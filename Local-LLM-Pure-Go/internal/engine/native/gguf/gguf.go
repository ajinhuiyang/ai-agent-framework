// Package gguf implements a pure-Go parser for the GGUF (GGML Unified Format) file format.
//
// GGUF v3 specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//
// File layout:
//
//	[Header]          magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
//	[Metadata KV]     repeated key-value pairs
//	[Tensor Infos]    repeated tensor descriptors (name, ndims, dims, type, offset)
//	[Padding]         alignment to ALIGNMENT boundary
//	[Tensor Data]     raw tensor data, each aligned to ALIGNMENT boundary
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sync"
)

const (
	MagicGGUF = 0x46554747 // "GGUF" as little-endian uint32 (bytes: 47 47 55 46)
	Alignment = 32         // Default data alignment
)

// GGUF value types (metadata).
const (
	TypeUint8   uint32 = 0
	TypeInt8    uint32 = 1
	TypeUint16  uint32 = 2
	TypeInt16   uint32 = 3
	TypeUint32  uint32 = 4
	TypeInt32   uint32 = 5
	TypeFloat32 uint32 = 6
	TypeBool    uint32 = 7
	TypeString  uint32 = 8
	TypeArray   uint32 = 9
	TypeUint64  uint32 = 10
	TypeInt64   uint32 = 11
	TypeFloat64 uint32 = 12
)

// GGML tensor data types.
const (
	GGMLTypeF32     uint32 = 0
	GGMLTypeF16     uint32 = 1
	GGMLTypeQ4_0    uint32 = 2
	GGMLTypeQ4_1    uint32 = 3
	GGMLTypeQ5_0    uint32 = 6
	GGMLTypeQ5_1    uint32 = 7
	GGMLTypeQ8_0    uint32 = 8
	GGMLTypeQ8_1    uint32 = 9
	GGMLTypeQ2_K    uint32 = 10
	GGMLTypeQ3_K    uint32 = 11
	GGMLTypeQ4_K    uint32 = 12
	GGMLTypeQ5_K    uint32 = 13
	GGMLTypeQ6_K    uint32 = 14
	GGMLTypeQ8_K    uint32 = 15
	GGMLTypeIQ2_XXS uint32 = 16
	GGMLTypeIQ2_XS  uint32 = 17
	GGMLTypeIQ3_XXS uint32 = 18
	GGMLTypeIQ1_S   uint32 = 19
	GGMLTypeIQ4_NL  uint32 = 20
	GGMLTypeIQ3_S   uint32 = 21
	GGMLTypeIQ2_S   uint32 = 22
	GGMLTypeIQ4_XS  uint32 = 23
	GGMLTypeBF16    uint32 = 30
)

// File represents a parsed GGUF file.
type File struct {
	Version     uint32
	TensorCount uint64
	Metadata    map[string]interface{}
	Tensors     []TensorInfo
	DataOffset  int64 // File offset where tensor data begins
	FilePath    string
}

// TensorInfo describes a single tensor in the GGUF file.
type TensorInfo struct {
	Name       string
	NDims      uint32
	Dimensions []uint64
	Type       uint32
	Offset     uint64 // Relative to data section start
}

// Parse reads and parses a GGUF file.
func Parse(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open gguf file: %w", err)
	}
	defer f.Close()

	return ParseReader(f, path)
}

// ParseReader parses a GGUF file from an io.ReadSeeker.
func ParseReader(r io.ReadSeeker, path string) (*File, error) {
	gf := &File{
		Metadata: make(map[string]interface{}),
		FilePath: path,
	}

	// Read header.
	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != MagicGGUF {
		return nil, fmt.Errorf("invalid magic: got 0x%08X, want 0x%08X (GGUF)", magic, MagicGGUF)
	}

	if err := binary.Read(r, binary.LittleEndian, &gf.Version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if gf.Version < 2 || gf.Version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d (supported: 2, 3)", gf.Version)
	}

	if err := binary.Read(r, binary.LittleEndian, &gf.TensorCount); err != nil {
		return nil, fmt.Errorf("read tensor count: %w", err)
	}

	var metadataKVCount uint64
	if err := binary.Read(r, binary.LittleEndian, &metadataKVCount); err != nil {
		return nil, fmt.Errorf("read metadata kv count: %w", err)
	}

	// Read metadata key-value pairs.
	for i := uint64(0); i < metadataKVCount; i++ {
		key, err := readString(r)
		if err != nil {
			return nil, fmt.Errorf("read metadata key %d: %w", i, err)
		}
		val, err := readValue(r)
		if err != nil {
			return nil, fmt.Errorf("read metadata value for key %q: %w", key, err)
		}
		gf.Metadata[key] = val
	}

	// Read tensor infos.
	gf.Tensors = make([]TensorInfo, gf.TensorCount)
	for i := uint64(0); i < gf.TensorCount; i++ {
		name, err := readString(r)
		if err != nil {
			return nil, fmt.Errorf("read tensor name %d: %w", i, err)
		}

		var ndims uint32
		if err := binary.Read(r, binary.LittleEndian, &ndims); err != nil {
			return nil, fmt.Errorf("read tensor ndims %d: %w", i, err)
		}

		dims := make([]uint64, ndims)
		for j := uint32(0); j < ndims; j++ {
			if err := binary.Read(r, binary.LittleEndian, &dims[j]); err != nil {
				return nil, fmt.Errorf("read tensor dim %d/%d: %w", i, j, err)
			}
		}

		var dtype uint32
		if err := binary.Read(r, binary.LittleEndian, &dtype); err != nil {
			return nil, fmt.Errorf("read tensor type %d: %w", i, err)
		}

		var offset uint64
		if err := binary.Read(r, binary.LittleEndian, &offset); err != nil {
			return nil, fmt.Errorf("read tensor offset %d: %w", i, err)
		}

		gf.Tensors[i] = TensorInfo{
			Name:       name,
			NDims:      ndims,
			Dimensions: dims,
			Type:       dtype,
			Offset:     offset,
		}
	}

	// Calculate data section offset (aligned to Alignment).
	pos, err := r.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("get current position: %w", err)
	}
	gf.DataOffset = alignOffset(pos, Alignment)

	return gf, nil
}

// GetString returns a string metadata value.
func (f *File) GetString(key string) (string, bool) {
	v, ok := f.Metadata[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// GetUint32 returns a uint32 metadata value.
func (f *File) GetUint32(key string) (uint32, bool) {
	v, ok := f.Metadata[key]
	if !ok {
		return 0, false
	}
	switch val := v.(type) {
	case uint32:
		return val, true
	case int32:
		return uint32(val), true
	case uint64:
		return uint32(val), true
	case int64:
		return uint32(val), true
	default:
		return 0, false
	}
}

// GetFloat32 returns a float32 metadata value.
func (f *File) GetFloat32(key string) (float32, bool) {
	v, ok := f.Metadata[key]
	if !ok {
		return 0, false
	}
	switch val := v.(type) {
	case float32:
		return val, true
	case float64:
		return float32(val), true
	default:
		return 0, false
	}
}

// GetStringArray returns a string array metadata value.
func (f *File) GetStringArray(key string) ([]string, bool) {
	v, ok := f.Metadata[key]
	if !ok {
		return nil, false
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil, false
	}
	strs := make([]string, 0, len(arr))
	for _, item := range arr {
		if s, ok := item.(string); ok {
			strs = append(strs, s)
		}
	}
	return strs, true
}

// FindTensor finds a tensor by name.
func (f *File) FindTensor(name string) (*TensorInfo, bool) {
	for i := range f.Tensors {
		if f.Tensors[i].Name == name {
			return &f.Tensors[i], true
		}
	}
	return nil, false
}

// TensorDataSize returns the byte size of a tensor's data.
func (t *TensorInfo) TensorDataSize() uint64 {
	elements := uint64(1)
	for _, d := range t.Dimensions {
		elements *= d
	}
	return elements * GGMLTypeSize(t.Type) / GGMLTypeBlockSize(t.Type)
}

// ReadTensorData reads the raw tensor data from the file.
func (f *File) ReadTensorData(t *TensorInfo) ([]byte, error) {
	file, err := os.Open(f.FilePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	offset := f.DataOffset + int64(t.Offset)
	if _, err := file.Seek(offset, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek to tensor data: %w", err)
	}

	size := t.TensorDataSize()
	data := make([]byte, size)
	if _, err := io.ReadFull(file, data); err != nil {
		return nil, fmt.Errorf("read tensor data: %w", err)
	}
	return data, nil
}

// ReadTensorFloat32 reads a tensor and dequantizes it to float32.
func (f *File) ReadTensorFloat32(t *TensorInfo) ([]float32, error) {
	data, err := f.ReadTensorData(t)
	if err != nil {
		return nil, err
	}
	return Dequantize(data, t.Type, t.ElementCount()), nil
}

// ElementCount returns the total number of elements in the tensor.
func (t *TensorInfo) ElementCount() uint64 {
	n := uint64(1)
	for _, d := range t.Dimensions {
		n *= d
	}
	return n
}

// --- Helper functions ---

func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 1<<20 { // Sanity check: 1MB max
		return "", fmt.Errorf("string too long: %d", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readValue(r io.Reader) (interface{}, error) {
	var vtype uint32
	if err := binary.Read(r, binary.LittleEndian, &vtype); err != nil {
		return nil, err
	}
	return readTypedValue(r, vtype)
}

func readTypedValue(r io.Reader, vtype uint32) (interface{}, error) {
	switch vtype {
	case TypeUint8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeInt8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeUint16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeInt16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeFloat32:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeBool:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	case TypeString:
		return readString(r)
	case TypeArray:
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, err
		}
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return nil, err
		}
		arr := make([]interface{}, count)
		for i := uint64(0); i < count; i++ {
			v, err := readTypedValue(r, elemType)
			if err != nil {
				return nil, err
			}
			arr[i] = v
		}
		return arr, nil
	case TypeUint64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeInt64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeFloat64:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	default:
		return nil, fmt.Errorf("unknown value type: %d", vtype)
	}
}

func alignOffset(offset int64, alignment int) int64 {
	a := int64(alignment)
	return (offset + a - 1) / a * a
}

// GGMLTypeSize returns bytes per block for a given GGML type.
func GGMLTypeSize(t uint32) uint64 {
	switch t {
	case GGMLTypeF32:
		return 4
	case GGMLTypeF16:
		return 2
	case GGMLTypeQ4_0:
		return 18 // 32 values: 2 bytes scale + 16 bytes data
	case GGMLTypeQ4_1:
		return 20 // 32 values: 2+2 bytes scale/min + 16 bytes data
	case GGMLTypeQ5_0:
		return 22
	case GGMLTypeQ5_1:
		return 24
	case GGMLTypeQ8_0:
		return 34 // 32 values: 2 bytes scale + 32 bytes data
	case GGMLTypeQ8_1:
		return 36
	case GGMLTypeQ2_K:
		return 84
	case GGMLTypeQ3_K:
		return 110
	case GGMLTypeQ4_K:
		return 144
	case GGMLTypeQ5_K:
		return 176
	case GGMLTypeQ6_K:
		return 210
	case GGMLTypeQ8_K:
		return 292
	case GGMLTypeBF16:
		return 2
	default:
		return 4 // fallback
	}
}

// GGMLTypeBlockSize returns the number of elements per block.
func GGMLTypeBlockSize(t uint32) uint64 {
	switch t {
	case GGMLTypeF32, GGMLTypeF16, GGMLTypeBF16:
		return 1
	case GGMLTypeQ4_0, GGMLTypeQ4_1, GGMLTypeQ5_0, GGMLTypeQ5_1,
		GGMLTypeQ8_0, GGMLTypeQ8_1:
		return 32
	case GGMLTypeQ2_K, GGMLTypeQ3_K, GGMLTypeQ4_K, GGMLTypeQ5_K,
		GGMLTypeQ6_K, GGMLTypeQ8_K:
		return 256
	default:
		return 1
	}
}

// GGMLTypeName returns a human-readable name for a GGML type.
func GGMLTypeName(t uint32) string {
	names := map[uint32]string{
		GGMLTypeF32: "F32", GGMLTypeF16: "F16", GGMLTypeQ4_0: "Q4_0",
		GGMLTypeQ4_1: "Q4_1", GGMLTypeQ5_0: "Q5_0", GGMLTypeQ5_1: "Q5_1",
		GGMLTypeQ8_0: "Q8_0", GGMLTypeQ8_1: "Q8_1", GGMLTypeQ2_K: "Q2_K",
		GGMLTypeQ3_K: "Q3_K", GGMLTypeQ4_K: "Q4_K", GGMLTypeQ5_K: "Q5_K",
		GGMLTypeQ6_K: "Q6_K", GGMLTypeQ8_K: "Q8_K", GGMLTypeBF16: "BF16",
	}
	if n, ok := names[t]; ok {
		return n
	}
	return fmt.Sprintf("Unknown(%d)", t)
}

// Dequantize converts quantized data to float32.
func Dequantize(data []byte, dtype uint32, elementCount uint64) []float32 {
	switch dtype {
	case GGMLTypeF32:
		return dequantF32(data, elementCount)
	case GGMLTypeF16:
		return dequantF16(data, elementCount)
	case GGMLTypeQ8_0:
		return dequantQ8_0(data, elementCount)
	case GGMLTypeQ4_0:
		return dequantQ4_0(data, elementCount)
	case GGMLTypeBF16:
		return dequantBF16(data, elementCount)
	default:
		// Fallback: zero-fill for unsupported types.
		out := make([]float32, elementCount)
		return out
	}
}

func dequantF32(data []byte, n uint64) []float32 {
	out := make([]float32, n)
	for i := uint64(0); i < n; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func dequantF16(data []byte, n uint64) []float32 {
	out := make([]float32, n)
	for i := uint64(0); i < n; i++ {
		bits := binary.LittleEndian.Uint16(data[i*2:])
		out[i] = float16ToFloat32(bits)
	}
	return out
}

func dequantBF16(data []byte, n uint64) []float32 {
	out := make([]float32, n)
	for i := uint64(0); i < n; i++ {
		bits := binary.LittleEndian.Uint16(data[i*2:])
		// BF16: top 16 bits of float32
		out[i] = math.Float32frombits(uint32(bits) << 16)
	}
	return out
}

// dequantQ8_0: block of 32 values, 2-byte f16 scale + 32 int8 values = 34 bytes.
// Parallelized for large tensors.
func dequantQ8_0(data []byte, n uint64) []float32 {
	out := make([]float32, n)
	const blockSize = 32
	const blockBytes = 34

	totalBlocks := (int(n) + blockSize - 1) / blockSize

	numWorkers := runtime.NumCPU()
	if totalBlocks < numWorkers {
		numWorkers = 1
	}
	blocksPerWorker := (totalBlocks + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		startBlock := w * blocksPerWorker
		endBlock := startBlock + blocksPerWorker
		if endBlock > totalBlocks {
			endBlock = totalBlocks
		}
		if startBlock >= endBlock {
			break
		}
		wg.Add(1)
		go func(sb, eb int) {
			defer wg.Done()
			for block := sb; block < eb; block++ {
				offset := block * blockBytes
				if offset+blockBytes > len(data) {
					break
				}
				scaleBits := binary.LittleEndian.Uint16(data[offset:])
				scale := float16ToFloat32(scaleBits)
				base := block * blockSize
				for j := 0; j < blockSize && base+j < int(n); j++ {
					out[base+j] = scale * float32(int8(data[offset+2+j]))
				}
			}
		}(startBlock, endBlock)
	}
	wg.Wait()
	return out
}

// dequantQ4_0: block of 32 values, 2-byte f16 scale + 16 bytes (4-bit pairs) = 18 bytes
func dequantQ4_0(data []byte, n uint64) []float32 {
	out := make([]float32, n)
	blockSize := 32
	blockBytes := 18

	for block := 0; block*blockSize < int(n); block++ {
		offset := block * blockBytes
		if offset+blockBytes > len(data) {
			break
		}
		scaleBits := binary.LittleEndian.Uint16(data[offset:])
		scale := float16ToFloat32(scaleBits)

		for j := 0; j < blockSize/2; j++ {
			b := data[offset+2+j]
			q0 := int(b&0x0F) - 8
			q1 := int(b>>4) - 8

			idx := block*blockSize + j*2
			if idx < int(n) {
				out[idx] = scale * float32(q0)
			}
			if idx+1 < int(n) {
				out[idx+1] = scale * float32(q1)
			}
		}
	}
	return out
}

// float16ToFloat32 converts IEEE 754 half-precision to single-precision.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	var f uint32
	switch {
	case exp == 0:
		if mant == 0 {
			f = sign << 31
		} else {
			// Subnormal
			exp = 1
			for mant&0x400 == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x3FF
			f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
		}
	case exp == 0x1F:
		// Inf or NaN
		f = (sign << 31) | (0xFF << 23) | (mant << 13)
	default:
		f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(f)
}
