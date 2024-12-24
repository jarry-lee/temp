package main

import (
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"image/color"
	"time"
)

func main() {
	// 애플리케이션 생성
	myApp := app.New()
	myWindow := myApp.NewWindow("Brightness Square")
	myWindow.Resize(fyne.NewSize(400, 400))

	// 밝기 초기값 설정
	brightness := 0

	// 정사각형 생성
	rect := canvas.NewRectangle(color.RGBA{R: 0, G: 0, B: 0, A: 255})
	rect.SetMinSize(fyne.NewSize(400, 400))

	// 타이머를 이용해 밝기 업데이트
	go func() {
		for {
			time.Sleep(30 * time.Millisecond) // 30ms 간격으로 밝기 변화
			brightness = (brightness + 1) % 256
			rgba := uint8(brightness)
			rect.FillColor = color.RGBA{R: rgba, G: rgba, B: rgba, A: 255}
			canvas.Refresh(rect) // 색상 업데이트
		}
	}()

	// 창에 정사각형 추가 및 실행
	myWindow.SetContent(rect)
	myWindow.ShowAndRun()
}
