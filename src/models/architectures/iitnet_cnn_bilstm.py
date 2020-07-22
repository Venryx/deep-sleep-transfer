# @sp: newly created

import keras
from keras import Input, models
from keras.layers import Conv1D, BatchNormalization, Activation, add, ZeroPadding1D, MaxPooling1D, Dropout, concatenate, \
    Bidirectional, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2


def iitnet_conv_identity_block(input_x,
                               filters,
                               line,
                               stage,
                               block,
                               kernel_size=3):
    """
    Implementation of the identity block (or bottleneck block) as described in
    the paper to ResNet50 (https://arxiv.org/pdf/1512.03385.pdf) and implemented
    in Keras. Altered as described in Back (2019), p.5

    :param input_x:     Input Tensor
    :param filters:     numbers of filters in the 3 convolutional layers (main path)
    :param line:        int, current line of input (one line for every input subsample)
    :param stage:       int, current stage (see ResNet50)
    :param block:       char, label of current block (see ResNet50)
    :param kernel_size: default 3, kernel size of the middle convolutional layer (main path)

    :return: Output Tensor
    """
    # define name basis for the layers in this block
    conv_name_base = 'line' + str(line) + '_res' + str(stage) + block + '_branch'
    bn_name_base = 'line' + str(line) + '_bn' + str(stage) + block + '_branch'

    # get the number of filters for the three convolutional layers
    filters1, filters2, filters3 = filters

    # store the input_tensor
    x_shortcut = input_x

    ## Build the main path
    # first component
    x = Conv1D(filters=filters1,
               kernel_size=1,
               strides=1,
               padding='valid',
               kernel_initializer='he_normal',
               name=conv_name_base+'2a')(input_x)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    # second component
    x = Conv1D(filters=filters2,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base+'2b')(x)
    x = BatchNormalization(name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    # third component
    x = Conv1D(filters=filters3,
               kernel_size=1,
               strides=1,
               padding='valid',
               kernel_initializer='he_normal',
               name=conv_name_base+'2c')(x)
    x = BatchNormalization(name=bn_name_base+'2c')(x)

    # join the main path and the recurrent shortcut
    x = add([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def iitnet_conv_convolutional_block(input_x,
                                    filters,
                                    line,
                                    stage,
                                    block,
                                    kernel_size=3,
                                    strides=2):
    """
    Implementation of the convolutional (or standard) block as described in
    the paper to ResNet50 (https://arxiv.org/pdf/1512.03385.pdf) and implemented
    in Keras. Altered as described in Back (2019), p.5

    :param input_x:     Input tensor
    :param filters:     numbers of filters in the 3 convolutional layers (main path)
    :param line:        int, current line of input (one line for every input subsample)
    :param stage:       int, current stage (see ResNet50)
    :param block:       char, label of current block (see ResNet50)
    :param kernel_size: default 3, kernel size of the middle convolutional layer (main path)
    :param strides:     strides for the first convolutional layer in both branches

    :return: Output Tensor
    """

    # define name basis for the layers in this block
    conv_name_base = 'line' + str(line) + '_res' + str(stage) + block + '_branch'
    bn_name_base = 'line' + str(line) + '_bn' + str(stage) + block + '_branch'

    # get the number of filters for the three convolutional layers
    filters1, filters2, filters3 = filters

    # store the input_tensor
    x_shortcut = input_x

    ## Build the main path
    # first component
    x = Conv1D(filters=filters1,
               kernel_size=1,
               strides=strides,
               kernel_initializer='he_normal',
               name=conv_name_base+'2a')(input_x)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    # second component
    x = Conv1D(filters=filters2,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base+'2b')(x)
    x = BatchNormalization(name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    # third component
    x = Conv1D(filters=filters3,
               kernel_size=1,
               strides=1,
               padding='valid',
               kernel_initializer='he_normal',
               name=conv_name_base+'2c')(x)
    x = BatchNormalization(name=bn_name_base+'2c')(x)

    ## Build the shortcut path
    x_shortcut = Conv1D(filters=filters3,
                        kernel_size=1,
                        strides=strides,
                        kernel_initializer='he_normal',
                        name=conv_name_base+'1')(x_shortcut)
    x_shortcut = BatchNormalization(name=bn_name_base+'1')(x_shortcut)

    ## Join the main path and the recurrent shortcut
    x = add([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def iitnet_convolutional_layers(x_input, line, first_filters):
    """
    Build the ResNet50 architecture as described in the paper to ResNet50
    (https://arxiv.org/pdf/1512.03385.pdf) and implemented in Keras. Altered
    as described in Back (2019), p.5

    :param x_input: Input tensor
    :param line:    int, current line of input (one line for every input subsample)
    :param first_filters: number of filters in the first stage

    :return: Output Tensor
    """

    line_index = str(line)

    # set the numbers of filters in the stages of the model
    stage1_filters = first_filters
    stage2_filters = [first_filters, first_filters, first_filters*2]
    stage3_filters = [first_filters*2, first_filters*2, first_filters * 4]
    stage4_filters = [first_filters * 4, first_filters * 4, first_filters * 8]
    stage5_filters = [first_filters * 8, first_filters * 8, first_filters * 16]

    # Zero Padding
    x = ZeroPadding1D(padding=3,
                      name='conv1_pad_line'+line_index)(x_input)

    # Stage 1
    x = Conv1D(filters=stage1_filters,
               kernel_size=2,
               padding='valid',
               kernel_initializer='he_normal',
               name='conv1_line'+line_index)(x)
    x = BatchNormalization(name='bn_conv1_line'+line_index)(x)
    x = Activation('relu')(x)
    x = ZeroPadding1D(padding=1,
                      name='pool1_pad_line'+line_index)(x)
    x = MaxPooling1D(pool_size=3,
                     strides=2)(x)

    # Stage 2
    x = iitnet_conv_convolutional_block(input_x=x,
                                        kernel_size=3,
                                        filters=stage2_filters,
                                        stage=2,
                                        block='a',
                                        line=line_index,
                                        strides=1)
    x = iitnet_conv_identity_block(input_x=x,
                                   kernel_size=3,
                                   filters=stage2_filters,
                                   stage=2,
                                   block='b',
                                   line=line_index)
    x = iitnet_conv_identity_block(input_x=x,
                                   kernel_size=3,
                                   filters=stage2_filters,
                                   stage=2,
                                   block='c',
                                   line=line_index)

    # Stage 3
    x = iitnet_conv_convolutional_block(input_x=x,
                                        kernel_size=3,
                                        filters=stage3_filters,
                                        stage=3,
                                        block='a',
                                        line=line_index)
    x = iitnet_conv_identity_block(input_x=x,
                                   kernel_size=3,
                                   filters=stage3_filters,
                                   stage=3,
                                   block='b',
                                   line=line_index)
    x = iitnet_conv_identity_block(input_x=x,
                                   kernel_size=3,
                                   filters=stage3_filters,
                                   stage=3,
                                   block='c',
                                   line=line_index)
    x = iitnet_conv_identity_block(input_x=x,
                                   kernel_size=3,
                                   filters=stage3_filters,
                                   stage=3,
                                   block='d',
                                   line=line_index)

    # Additional Pooling to reduce the length of the feature sequence by half
    x = MaxPooling1D(pool_size=2)(x)

    # Stage 4
    x = iitnet_conv_convolutional_block(input_x=x,
                                        kernel_size=3,
                                        filters=stage4_filters,
                                        stage=4,
                                        block='a',
                                        line=line_index)
    x = iitnet_conv_identity_block(input_x=x,
                                   kernel_size=3,
                                   filters=stage4_filters,
                                   stage=4,
                                   block='b',
                                   line=line_index)
    x = iitnet_conv_identity_block(input_x=x,
                                   kernel_size=3,
                                   filters=stage4_filters,
                                   stage=4,
                                   block='c',
                                   line=line_index)
    x = iitnet_conv_identity_block(input_x=x,
                                   kernel_size=3,
                                   filters=stage4_filters,
                                   stage=4,
                                   block='d',
                                   line=line_index)
    x = iitnet_conv_identity_block(input_x=x,
                                   kernel_size=3,
                                   filters=stage4_filters,
                                   stage=4,
                                   block='e',
                                   line=line_index)
    x = iitnet_conv_identity_block(input_x=x,
                                   kernel_size=3,
                                   filters=stage4_filters,
                                   stage=4,
                                   block='f',
                                   line=line_index)

    #Stage 5
    x = iitnet_conv_convolutional_block(input_x=x,
                                        kernel_size=3,
                                        filters=stage5_filters,
                                        stage=5,
                                        block='a',
                                        line=line_index)
    x = iitnet_conv_identity_block(input_x=x,
                                   kernel_size=3,
                                   filters=stage5_filters,
                                   stage=5,
                                   block='b',
                                   line=line_index)

    x = iitnet_conv_identity_block(input_x=x,
                                   kernel_size=3,
                                   filters=stage5_filters,
                                   stage=5,
                                   block='c',
                                   line=line_index)

    # Dropout layer to prevent overfitting
    x = Dropout(0.5)(x)

    return x


def build_model_iitnet(params):
    """
    Construct the IITNet architecture as described in Back (2019)

    :param params: contains all parameters for model building

    :return: Keras classification model
    """

    # set parameters from params-object
    l_epochs = params.plx.get('l_epochs')
    l_subepochs = params.plx.get('l_subepochs')
    l_overlapping = params.plx.get('l_overlapping')
    n_channels = len(params.plx.get('ch_idx_list'))
    epoch_length_sec = params.plx.get('sections')
    n_classes = len(params.plx.get('key_labels'))
    n_filters = params.plx.get('filtersize')
    if params.plx.get('apply_downsampling'):
        frequency = params.plx.get('common_frequency')
    else:
        frequency = params.plx.get('frequency')

    # get the number of all subepochs in the subsample
    n_subepochs_input = l_epochs * l_subepochs

    # get the length of one subepoch (in seconds and in datapoints)
    subepoch_length_seconds = epoch_length_sec / l_subepochs
    subepoch_length_points = subepoch_length_seconds * frequency
    # subepochs overlap, so they have to be longer by l percent
    # overlapping can't be applied if there is only one subepoch per epoch
    if l_subepochs > 1:
        subepoch_length_points = int(subepoch_length_points * (1 + (l_overlapping/100)))

    # define the shape of all model inputs (= shape of the subepochs)
    input_shape = (int(subepoch_length_points), n_channels)

    # prepare list of feature sequences from all the input lines
    x_list = []

    # make a list of names for the input layers
    names_input_layers = []
    for i in range(n_subepochs_input):
        name = "x_input_" + str(i)
        names_input_layers.append(name)

    # create dictionary input layers
    input_layers = {}
    for i, input_layer in enumerate(names_input_layers):
        # create first layer in the line to represent the input data
        input_layers[input_layer] = Input(input_shape)

        # calculate the feature sequence for this subepoch
        x = iitnet_convolutional_layers(input_layers[input_layer], line=i, first_filters=n_filters)

        x_list.append(x)

    # bring the lines together
    if n_subepochs_input > 1:
        x = concatenate(x_list)
    else:
        x = x_list[0]

    # RECURRENT LAYERS
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128))(x)

    # CLASSIFICATION LAYER
    x = Dense(n_classes,
              activation='softmax',
              name='fc'+str(n_classes))(x)

    # CREATE MODEL
    model = Model(list(input_layers.values()), x, name='IITNet')

    return model


def add_regularization(model, params):
    """
    Add regularization, specified in the parameters, to all layers of the model.
    see also https://sthalles.github.io/keras-regularizer/
    :param model:   built iitnet model
    :param params:
    """
    l2_reg_factor = params.plx.get('l2_regularization_factor')
    l2_regularizer = l2(l=l2_reg_factor)

    # set this regularizer for every layer that can have a regularizer
    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, l2_regularizer)

    # check the set losses
    model_json = model.to_json()
    model = models.model_from_json(model_json)

    print(model.losses)

    return model


def compile_model_iitnet(params, model):
    """
    Compile an IITNet model, for more details see paper from Back (2019)
    :param params: Params-Object, contains parameters for model building
    :param model:  Keras-Model
    :return: Keras-Model
    """

    learning_rate = params.plx.get('lr')
    beta1 = params.plx.get('beta1')
    beta2 = params.plx.get('beta2')

    adam_optimizer = Adam(lr=learning_rate,
                          beta_1=beta1,
                          beta_2=beta2)

    # model_gpu = keras.utils.multi_gpu_model(model, gpus=2)
    # model_gpu.compile(loss='categorical_crossentropy',
    #                   optimizer=adam_optimizer,
    #                   metrics=['accuracy'])

    model_gpu = model.compile(loss='categorical_crossentropy',
                      optimizer=adam_optimizer,
                      metrics=['accuracy'])

    return model_gpu


def create_model_iitnet(params, model, compile=True):
    """
    Build and compile an IITNet model, for more details see paper from Back (2019)
    :param params: Params-Object, contains parameters for model building
    :param model:  Keras-Model
    :return: Keras-Model
    """

    if model is None:
        # create model
        print("Building IITNet ...")
        model = build_model_iitnet(params)
        # add regularization to the whole model
        model = add_regularization(model, params)

        if compile:
            print("\nCompiling model ... ", end=" ")
            compile_model_iitnet(params, model)
            print("done")

        model.summary()

    return model
