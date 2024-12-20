package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"os"
)

// WAV 헤더 구조체
type WAVHeader struct {
	ChunkID       [4]byte
	ChunkSize     uint32
	Format        [4]byte
	Subchunk1ID   [4]byte
	Subchunk1Size uint32
	AudioFormat   uint16
	NumChannels   uint16
	SampleRate    uint32
	ByteRate      uint32
	BlockAlign    uint16
	BitsPerSample uint16
	Subchunk2ID   [4]byte
	Subchunk2Size uint32
}

// 8PSK 심볼 매핑 테이블
var phaseTable = []complex128{
	1 + 0i,           // 000
	0.707 + 0.707i,   // 001
	0 + 1i,           // 010
	-0.707 + 0.707i,  // 011
	-1 + 0i,          // 100
	-0.707 - 0.707i,  // 101
	0 - 1i,           // 110
	0.707 - 0.707i,   // 111
}

// 바이너리 데이터를 8PSK 심볼로 변환
func binaryToSymbols(data []byte) []complex128 {
	var symbols []complex128
	for _, b := range data {
		for i := 0; i < 8; i += 3 {
			index := (b >> (5 - i)) & 0b111 // 3비트 추출
			symbols = append(symbols, phaseTable[index])
		}
	}
	return symbols
}

// 8PSK 심볼을 PCM 샘플로 변환
func symbolsToSamples(symbols []complex128, sampleRate int, toneFrequency float64, duration float64) []float64 {
	var samples []float64
	samplesPerSymbol := int(float64(sampleRate) * duration)

	for _, symbol := range symbols {
		phase := cmplx.Phase(symbol)
		for i := 0; i < samplesPerSymbol; i++ {
			t := float64(i) / float64(sampleRate)
			sample := math.Sin(2*math.Pi*toneFrequency + phase)
			samples = append(samples, sample)
		}
	}
	return samples
}

// WAV 파일 생성
func createWAVFile(filePath string, sampleRate int, samples []float64) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	// WAV 헤더 작성
	numSamples := len(samples)
	header := WAVHeader{
		ChunkID:       [4]byte{'R', 'I', 'F', 'F'},
		ChunkSize:     36 + uint32(numSamples*2),
		Format:        [4]byte{'W', 'A', 'V', 'E'},
		Subchunk1ID:   [4]byte{'f', 'm', 't', ' '},
		Subchunk1Size: 16,
		AudioFormat:   1,
		NumChannels:   1,
		SampleRate:    uint32(sampleRate),
		ByteRate:      uint32(sampleRate * 2),
		BlockAlign:    2,
		BitsPerSample: 16,
		Subchunk2ID:   [4]byte{'d', 'a', 't', 'a'},
		Subchunk2Size: uint32(numSamples * 2),
	}

	// 헤더 저장
	err = binary.Write(file, binary.LittleEndian, &header)
	if err != nil {
		return err
	}

	// 샘플 데이터 저장
	for _, sample := range samples {
		intSample := int16(sample * math.MaxInt16)
		err = binary.Write(file, binary.LittleEndian, intSample)
		if err != nil {
			return err
		}
	}

	return nil
}

// WAV 파일에서 PCM 샘플 읽기
func readWAVFile(filePath string) ([]float64, int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()

	var header WAVHeader
	err = binary.Read(file, binary.LittleEndian, &header)
	if err != nil {
		return nil, 0, err
	}

	// PCM 데이터 읽기
	var samples []float64
	for {
		var intSample int16
		err = binary.Read(file, binary.LittleEndian, &intSample)
		if err != nil {
			break
		}
		samples = append(samples, float64(intSample)/math.MaxInt16)
	}

	return samples, int(header.SampleRate), nil
}

// PCM 샘플을 8PSK 심볼로 디코딩
func samplesToSymbols(samples []float64, sampleRate int, toneFrequency float64, duration float64) []complex128 {
	samplesPerSymbol := int(float64(sampleRate) * duration)
	var symbols []complex128

	for i := 0; i < len(samples); i += samplesPerSymbol {
		var sum complex128
		for j := 0; j < samplesPerSymbol && i+j < len(samples); j++ {
			t := float64(j) / float64(sampleRate)
			sum += complex(samples[i+j], 0) * cmplx.Exp(complex(0, -2*math.Pi*toneFrequency*t))
		}
		symbols = append(symbols, sum)
	}

	return symbols
}

// 8PSK 심볼을 바이너리 데이터로 변환
func symbolsToBinary(symbols []complex128) []byte {
	var data []byte
	var currentByte byte
	var bitCount int

	for _, symbol := range symbols {
		closest := 0
		minDist := cmplx.Abs(symbol - phaseTable[0])
		for i := 1; i < len(phaseTable); i++ {
			dist := cmplx.Abs(symbol - phaseTable[i])
			if dist < minDist {
				closest = i
				minDist = dist
			}
		}

		currentByte = (currentByte << 3) | byte(closest)
		bitCount += 3
		if bitCount >= 8 {
			data = append(data, currentByte)
			currentByte = 0
			bitCount = 0
		}
	}

	return data
}

func main() {
	inputFile := "input.bin"
	outputWAV := "output.wav"
	decodedFile := "decoded.bin"

	// 1. 바이너리 파일 읽기
	file, err := os.Open(inputFile)
	if err != nil {
		log.Fatalf("파일 열기 실패: %v", err)
	}
	defer file.Close()
	reader := bufio.NewReader(file)

	// 스트리밍 방식으로 처리
	var symbols []complex128
	buffer := make([]byte, 1024)
	for {
		n, err := reader.Read(buffer)
		if n == 0 {
			break
		}
		symbols = append(symbols, binaryToSymbols(buffer[:n])...)
		if err != nil {
			break
		}
	}

	// 2. 심볼 -> WAV 샘플 생성
	sampleRate := 44100
	toneFrequency := 1000.0
	duration := 0.01
	samples := symbolsToSamples(symbols, sampleRate, toneFrequency, duration)

	// 3. WAV 파일 생성
	err = createWAVFile(outputWAV, sampleRate, samples)
	if err != nil {
		log.Fatalf("WAV 파일 생성 실패: %v", err)
	}

	// 4. WAV 파일 읽기 및 디코딩
	readSamples, _, err := readWAVFile(outputWAV)
	if err != nil {
		log.Fatalf("WAV 파일 읽기 실패: %v", err)
	}
	decodedSymbols := samplesToSymbols(readSamples, sampleRate, toneFrequency, duration)
	decodedData := symbolsToBinary(decodedSymbols)

	// 5. 디코딩된 바이너리 파일 저장
	outFile, err := os.Create(decodedFile)
	if err != nil {
		log.Fatalf("파일 생성 실패: %v", err)
	}
	defer outFile.Close()
	outFile.Write(decodedData)

	log.Println("인코딩 및 디코딩 완료.")
}
