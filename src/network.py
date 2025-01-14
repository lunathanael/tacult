import sys
sys.path.append('..')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Flatten, Dense, Add, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy
import tensorflow as tf

class UtacNNet():
    def __init__(self, args):
        # Enable mixed precision training
        if args.cuda:
            set_global_policy('mixed_float16')
            
        # Configure GPU memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices and args.cuda:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        # game params
        self.board_x, self.board_y, self.board_z = 9, 9, 2
        self.action_size = 81
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y, self.board_z))

        # Convolution layers
        x = Conv2D(128, 3, padding='same')(self.input_boards)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Multiple residual blocks to process board patterns
        for _ in range(8):
            x = self.residual_block(x, 128)

        # Final convolution to increase channels
        x = Conv2D(256, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        # Flatten for policy and value heads
        x = Flatten()(x)

        # Policy head
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(x)

        # Value head
        self.v = Dense(256, activation='relu')(x)
        self.v = Dense(1, activation='tanh', name='v')(self.v)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'], 
            optimizer=Adam(args.lr)
        )

    def residual_block(self, x, filters):
        """Simple residual block"""
        y = Conv2D(filters, 3, padding='same')(x)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = Conv2D(filters, 3, padding='same')(y)
        y = BatchNormalization()(y)
        out = Add()([x, y])
        return LeakyReLU()(out)