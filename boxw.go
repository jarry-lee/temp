package main

import (
	"github.com/lxn/walk"
	"github.com/lxn/win"
	"image/color"
	"time"
)

func main() {
	// 새로운 애플리케이션과 윈도우 생성
	mainWindow, _ := walk.NewMainWindow()

	// 패널 생성
	panel, _ := walk.NewComposite(mainWindow)

	// 패널 크기 설정
	panel.SetBackground(walk.SolidColorBrush{Color: walk.RGB(0, 0, 0)})

	// 창 크기 설정
	mainWindow.SetTitle("Brightness Square")
	mainWindow.SetSize(walk.Size{Width: 400, Height: 400})

	// 밝기 초기값
	brightness := 0

	// 타이머를 이용한 밝기 업데이트
	go func() {
		for {
			time.Sleep(30 * time.Millisecond) // 30ms 간격으로 밝기 변화
			brightness = (brightness + 1) % 256
			rgba := uint8(brightness)
			color := color.RGBA{R: rgba, G: rgba, B: rgba, A: 255}

			// Windows API를 사용해 패널 배경 변경
			winColor := win.COLORREF(uint32(color.B)<<16 | uint32(color.G)<<8 | uint32(color.R))
			win.SetBkColor(win.GetDC(win.HWND(panel.Handle())), winColor)

			// 새로고침
			panel.Invalidate()
		}
	}()

	// 창 실행
	mainWindow.Run()
}
