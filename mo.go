package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"math"
	"os"
)

// 8PSK 심볼 매핑
var symbolTable = []complex128{
	1 + 0i,                      // 000
	0.707 + 0.707i,              // 001
	0 + 1i,                      // 010
	-0.707 + 0.707i,             // 011
	-1 + 0i,                     // 100
	-0.707 - 0.707i,             // 101
	0 - 1i,                      // 110
	0.707 - 0.707i,              // 111
}

// 인코딩: 3비트를 8PSK 심볼로 변환
func encode8PSK(input []byte) []complex128 {
	var encoded []complex128
	for _, b := range input {
		for i := 0; i < 8; i += 3 { // 3비트씩 처리
			idx := (b >> (5 - i)) & 0x07 // 상위 3비트 추출
			encoded = append(encoded, symbolTable[idx])
		}
	}
	return encoded
}

// 디코딩: 8PSK 심볼을 3비트 데이터로 변환
func decode8PSK(encoded []complex128) []byte {
	var decoded []byte
	var currentByte byte
	var bitCount int

	for _, symbol := range encoded {
		// 가장 가까운 심볼 찾기
		closestIdx := 0
		minDist := math.MaxFloat64
		for i, ref := range symbolTable {
			dist := cmplx.Abs(symbol - ref)
			if dist < minDist {
				minDist = dist
				closestIdx = i
			}
		}

		// 3비트 추가
		currentByte = (currentByte << 3) | byte(closestIdx)
		bitCount += 3

		// 8비트가 채워지면 저장
		if bitCount >= 8 {
			decoded = append(decoded, currentByte)
			currentByte = 0
			bitCount = 0
		}
	}

	return decoded
}

// Hamming Code (7,4) 인코딩
func hammingEncode(input byte) byte {
	d0 := (input >> 3) & 1
	d1 := (input >> 2) & 1
	d2 := (input >> 1) & 1
	d3 := input & 1

	p0 := d0 ^ d1 ^ d3
	p1 := d0 ^ d2 ^ d3
	p2 := d1 ^ d2 ^ d3

	return (p0 << 6) | (p1 << 5) | (p2 << 4) | (d0 << 3) | (d1 << 2) | (d2 << 1) | d3
}

// Hamming Code (7,4) 디코딩 및 오류 수정
func hammingDecode(encoded byte) (byte, bool) {
	d0 := (encoded >> 3) & 1
	d1 := (encoded >> 2) & 1
	d2 := (encoded >> 1) & 1
	d3 := encoded & 1

	p0 := (encoded >> 6) & 1
	p1 := (encoded >> 5) & 1
	p2 := (encoded >> 4) & 1

	s0 := p0 ^ d0 ^ d1 ^ d3
	s1 := p1 ^ d0 ^ d2 ^ d3
	s2 := p2 ^ d1 ^ d2 ^ d3

	syndrome := (s0 << 2) | (s1 << 1) | s2

	if syndrome != 0 {
		errorPosition := 7 - syndrome // 오류 위치 계산
		encoded ^= 1 << errorPosition
	}

	return (d0 << 3) | (d1 << 2) | (d2 << 1) | d3, syndrome != 0
}

// 스트림 방식으로 파일 인코딩
func encodeFile(inputPath, outputPath string) error {
	inputFile, err := os.Open(inputPath)
	if err != nil {
		return fmt.Errorf("입력 파일 열기 실패: %v", err)
	}
	defer inputFile.Close()

	outputFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("출력 파일 생성 실패: %v", err)
	}
	defer outputFile.Close()

	reader := bufio.NewReader(inputFile)
	writer := bufio.NewWriter(outputFile)

	buffer := make([]byte, 1024)
	for {
		n, err := reader.Read(buffer)
		if err != nil && err != io.EOF {
			return fmt.Errorf("파일 읽기 실패: %v", err)
		}
		if n == 0 {
			break
		}

		// Hamming Code 인코딩
		for i := 0; i < n; i++ {
			buffer[i] = hammingEncode(buffer[i])
		}

		// 8PSK 인코딩
		encoded := encode8PSK(buffer[:n])

		// 복소수를 바이트 스트림으로 변환
		for _, sym := range encoded {
			_, err := writer.Write([]byte(fmt.Sprintf("%f,%f\n", real(sym), imag(sym))))
			if err != nil {
				return fmt.Errorf("파일 쓰기 실패: %v", err)
			}
		}
	}

	writer.Flush()
	return nil
}

// 스트림 방식으로 파일 디코딩
func decodeFile(inputPath, outputPath string) error {
	inputFile, err := os.Open(inputPath)
	if err != nil {
		return fmt.Errorf("입력 파일 열기 실패: %v", err)
	}
	defer inputFile.Close()

	outputFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("출력 파일 생성 실패: %v", err)
	}
	defer outputFile.Close()

	reader := bufio.NewReader(inputFile)
	writer := bufio.NewWriter(outputFile)

	var encoded []complex128
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil && err != io.EOF {
			return fmt.Errorf("파일 읽기 실패: %v", err)
		}
		if len(line) == 0 {
			break
		}

		var re, im float64
		_, err = fmt.Sscanf(string(line), "%f,%f", &re, &im)
		if err != nil {
			return fmt.Errorf("데이터 파싱 실패: %v", err)
		}
		encoded = append(encoded, complex(re, im))
	}

	// 8PSK 디코딩
	decoded := decode8PSK(encoded)

	// Hamming Code 디코딩 및 오류 수정
	for i := 0; i < len(decoded); i++ {
		decoded[i], _ = hammingDecode(decoded[i])
	}

	// 디코딩된 데이터를 파일로 저장
	_, err = writer.Write(decoded)
	if err != nil {
		return fmt.Errorf("파일 쓰기 실패: %v", err)
	}

	writer.Flush()
	return nil
}

func main() {
	// 파일 경로 설정
	inputFile := "input.txt"
	encodedFile := "encoded.txt"
	decodedFile := "decoded.txt"

	// 인코딩
	err := encodeFile(inputFile, encodedFile)
	if err != nil {
		log.Fatalf("인코딩 실패: %v", err)
	}
	fmt.Println("인코딩 완료:", encodedFile)

	// 디코딩
	err = decodeFile(encodedFile, decodedFile)
	if err != nil {
		log.Fatalf("디코딩 실패: %v", err)
	}
	fmt.Println("디코딩 완료:", decodedFile)
}
