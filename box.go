package main

import (
    "github.com/AllenDang/giu"
    "github.com/AllenDang/giu/imgui"
    "image/color"
    "time"
)

var brightness int

func loop() {
    // 밝기 조정
    brightness = (brightness + 1) % 256
    rgba := uint8(brightness)

    // 창에 그릴 내용 설정
    g := giu.NewMasterWindow("Brightness Square", 400, 400, 0)
    g.Run(func() {
        giu.SingleWindow().Layout(
            giu.Custom(func() {
                imgui.GetWindowDrawList().AddRectFilled(
                    imgui.Vec2{X: 50, Y: 50},
                    imgui.Vec2{X: 350, Y: 350},
                    imgui.ColorConvertRGBAToVec4(color.RGBA{R: rgba, G: rgba, B: rgba, A: 255}),
                )
            }),
        )
    })
}

func main() {
    // 60 FPS로 실행
    giu.NewMasterWindow("Brightness Demo", 400, 400, giu.MasterWindowFlagsNotResizable).Run(loop)
}
