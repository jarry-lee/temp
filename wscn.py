import tensorflow as tf
from tensorflow.keras import layers, models, Input, backend as K
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. 사용자 정의 손실 함수: BCE-Dice Loss
# -------------------------------------------------------------------------
def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = 1.0 - dice_coefficient(y_true, y_pred)
    return bce + dice

# -------------------------------------------------------------------------
# 2. WSCN 모델 아키텍처 정의 (1024x1024 RGB 입력용)
# -------------------------------------------------------------------------
def wscn_conv_block(x, filters, kernel_size=(3, 3), padding='same', use_separable=False):
    """WSCN 기본 컨볼루션 블록: Conv -> BN -> Activation -> Dropout -> (Sep)Conv -> BN -> Activation"""
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    if use_separable:
        x = layers.SeparableConv2D(filters, kernel_size, padding=padding)(x)
    else:
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def wscn_decoder_block(x, skip_features, filters, kernel_size=(3, 3), padding='same'):
    """Decoder 블록: Upsampling -> Concatenate -> Conv Block"""
    # 1. Upsampling (Transpose Conv를 통해 크기 2배 확대)
    x = layers.Conv2DTranspose(filters, kernel_size, strides=(2, 2), padding=padding)(x)
    
    # 2. Skip Connection (Encoder의 특징맵과 결합)
    if skip_features is not None:
        shape_x = K.int_shape(x)          # (Batch, H, W, C)
        shape_skip = K.int_shape(skip_features)
        
        # [수정] 인덱스를 6, 7에서 1(H), 2(W)로 변경
        if shape_x[1] != shape_skip[1] or shape_x[2] != shape_skip[2]:
             # 크기가 다를 경우 resize 수행
             skip_features = tf.image.resize(skip_features, (shape_x[1], shape_x[2]))
        
        x = layers.Concatenate()([x, skip_features])
    
    # 3. Convolution
    x = wscn_conv_block(x, filters, kernel_size, padding, use_separable=False)
    return x

def build_wscn_rgb_1024(input_shape=(1024, 1024, 3)):
    inputs = Input(shape=input_shape)

    # --- [Encoder] Feature Extraction (Downsampling) ---
    e1 = wscn_conv_block(inputs, 16, use_separable=True)
    p1 = layers.MaxPooling2D((2, 2))(e1)

    e2 = wscn_conv_block(p1, 32, use_separable=True)
    p2 = layers.MaxPooling2D((2, 2))(e2)

    e3 = wscn_conv_block(p2, 64, use_separable=True)
    p3 = layers.MaxPooling2D((2, 2))(e3)

    e4 = wscn_conv_block(p3, 128, use_separable=True)
    p4 = layers.MaxPooling2D((2, 2))(e4)

    # Bridge
    b1 = wscn_conv_block(p4, 256, use_separable=False)

    # --- [Decoder] Segmentation Branch (Upsampling) ---
    d1 = wscn_decoder_block(b1, e4, 128)
    d2 = wscn_decoder_block(d1, e3, 64)
    d3 = wscn_decoder_block(d2, e2, 32)
    d4 = wscn_decoder_block(d3, e1, 16)

    # --- Output Layer ---
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', name='defect_mask')(d4)

    model = models.Model(inputs=inputs, outputs=outputs, name="WSCN_RGB_1024_Segmentation")
    return model

# -------------------------------------------------------------------------
# 3. 메인 실행 블록
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # A. 모델 생성
    print("1. 모델을 생성하는 중...")
    input_shape = (1024, 1024, 3) 
    model = build_wscn_rgb_1024(input_shape)
    
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coefficient, 'accuracy'])

    # B. 더미 데이터 생성 (시뮬레이션용)
    print("\n2. 학습용 더미 데이터 생성 중...")
    num_samples = 5
    X_train = np.random.rand(num_samples, 1024, 1024, 3).astype(np.float32)
    Y_train = np.random.randint(0, 2, size=(num_samples, 1024, 1024, 1)).astype(np.float32)

    # C. 모델 학습
    print("\n3. 모델 학습 시작...")
    # 메모리 확보를 위해 batch_size=1 유지
    history = model.fit(X_train, Y_train, batch_size=1, epochs=1, verbose=1)

    # D. 예측 및 결과 시각화
    print("\n4. 결과 예측 및 시각화 생성...")
    test_img = X_train[0:1] # shape (1, 1024, 1024, 3)
    prediction = model.predict(test_img) 
    predicted_mask = (prediction > 0.5).astype(np.float32)

    # [수정] 시각화 단계에서 4차원 텐서를 3차원/2차원으로 변환
    plt.figure(figsize=(15, 5))
    
    # (1) 원본 이미지: (1, 1024, 1024, 3) -> (1024, 1024, 3)
    plt.subplot(1, 3, 1)
    plt.imshow(test_img[0]) # [0] 인덱싱으로 배치 차원 제거
    plt.title("Input RGB Wafer Image")
    plt.axis('off')

    # (2) 실제 정답 (Ground Truth): (1, 1024, 1024, 1) -> (1024, 1024)
    plt.subplot(1, 3, 2)
    plt.imshow(Y_train[0, :, :, 0], cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    # (3) 모델 예측 결과: (1, 1024, 1024, 1) -> (1024, 1024)
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask[0, :, :, 0], cmap='gray')
    plt.title("Predicted Defect Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    print("\n[완료] 모델 수정 및 시각화가 정상적으로 종료되었습니다.")
