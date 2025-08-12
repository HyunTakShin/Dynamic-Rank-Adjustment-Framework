'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2025/08/11
Hyuntak Shin, B.A Participant.
<hyuntakshin@inha.ac.kr>
'''
import tensorflow as tf

class LoRAConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 kernel_regularizer,
                 rank=4,
                 strides=(1,1),
                 padding='same',
                 use_bias=False,
                 alpha=16.0):
        super(LoRAConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.regularizer = kernel_regularizer

    def build(self, input_shape):
        in_channels = input_shape[-1]

        self.conv = tf.keras.layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            trainable=False  # 원 weight는 freeze
        )

        self.lora_down = tf.keras.layers.Conv2D(
            self.rank,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
            trainable=True
        )

        self.lora_up = tf.keras.layers.Conv2D(
            self.filters,
            kernel_size=1,
            strides=1,
            padding='valid',
            use_bias=False,
            kernel_regularizer = self.regularizer,
            trainable=True
        )

    def call(self, x):
        orig_out = self.conv(x)
        lora_out = self.lora_up(self.lora_down(x)) * self.scaling
        return orig_out + lora_out

class TuckerLoRAConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 kernel_regularizer,
                 rank=4,
                 strides=(1,1),
                 padding='same',
                 use_bias=False,
                 alpha=1.0):
        super(TuckerLoRAConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.regularizer = kernel_regularizer
        self.tucker_lora = True

    def build(self, input_shape):
        in_channels = input_shape[-1]

        self.conv = tf.keras.layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            trainable=False  # 원 weight는 freeze
        )

        self.lora_param1 = tf.keras.layers.Conv2D(
            self.rank,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
            trainable=True
        )

        self.lora_core = tf.keras.layers.Conv2D(
            self.rank,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
            trainable=True
        )

        self.lora_param2 = tf.keras.layers.Conv2D(
            self.filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
            trainable=True
        )

    def call(self, x):
        orig_out = self.conv(x)
        lora_out = self.lora_param2(self.lora_core(self.lora_param1(x))) * self.scaling
        return orig_out + lora_out

class CPLoRAConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 kernel_regularizer,
                 rank=4,
                 strides=(1,1),
                 padding='same',
                 use_bias=False,
                 alpha=64.0):
        super(CPLoRAConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.regularizer = kernel_regularizer
        self.cp_lora = True

    def build(self, input_shape):
        in_channels = input_shape[-1]

        self.conv = tf.keras.layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            trainable=False  # 원 weight는 freeze
        )

        self.cp1 = tf.keras.layers.Conv2D(
            self.rank,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
        )

        self.cp2 = tf.keras.layers.Conv2D(
            self.rank,
            kernel_size=(1,self.kernel_size[0]),
            strides=1,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
        )

        self.cp3 = tf.keras.layers.Conv2D(
            self.rank,
            kernel_size=(self.kernel_size[0],1),
            strides=1,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
        )

        self.cp4 = tf.keras.layers.Conv2D(
            self.filters,
            kernel_size=1,
            strides=self.strides,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
        )

    def call(self, x):
        orig_out = self.conv(x)
        lora_out = self.cp4(self.cp3(self.cp2(self.cp1(x)))) * self.scaling
        return orig_out + lora_out

class LowRankConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 kernel_regularizer,
                 kernel_initializer = None,
                 rank=4,
                 strides=(1,1),
                 padding='same',
                 use_bias=False):
        super(LowRankConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.rank = rank
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.initializer = kernel_initializer
        self.regularizer = kernel_regularizer

    def build(self, input_shape):
        in_channels = input_shape[-1]

        self.down = tf.keras.layers.Conv2D(
            self.rank,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            use_bias=False,
            kernel_initializer = self.initializer,
            kernel_regularizer = self.regularizer,
        )

        self.up = tf.keras.layers.Conv2D(
            self.filters,
            kernel_size=1,
            strides=1,
            padding='valid',
            use_bias=False,
            kernel_initializer = self.initializer,
            kernel_regularizer = self.regularizer,
        )

    def call(self, x):
        out = self.up(self.down(x))
        return out

class CPConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 kernel_regularizer,
                 rank=4,
                 strides=(1,1),
                 padding='same',
                 use_bias=False):
        super(CPConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.rank = rank
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.regularizer = kernel_regularizer
        self.cp = True

    def build(self, input_shape):
        self.cp1 = tf.keras.layers.Conv2D(
            self.rank,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
        )

        self.cp2 = tf.keras.layers.Conv2D(
            self.rank,
            kernel_size=(1,self.kernel_size[0]),
            strides=1,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
        )

        self.cp3 = tf.keras.layers.Conv2D(
            self.rank,
            kernel_size=(self.kernel_size[0],1),
            strides=1,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
        )

        self.cp4 = tf.keras.layers.Conv2D(
            self.filters,
            kernel_size=1,
            strides=self.strides,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
        )

    def call(self, x):
        out = self.cp1(x)
        out = self.cp2(out)
        out = self.cp3(out)
        out = self.cp4(out)
        return out

class TuckerConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 kernel_regularizer,
                 rank=4,
                 strides=(1, 1),
                 padding='same',
                 use_bias=False):
        super(TuckerConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.rank = rank
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.regularizer = kernel_regularizer

    def build(self, input_shape):
        in_channels = input_shape[-1]

        self.down = tf.keras.layers.Conv2D(
            self.rank,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
        )

        self.core = tf.keras.layers.Conv2D(
            self.rank,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
        )

        self.up = tf.keras.layers.Conv2D(
            self.filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_regularizer = self.regularizer,
        )

    def call(self, x):
        out = self.up(self.core(self.down(x)))
        return out
