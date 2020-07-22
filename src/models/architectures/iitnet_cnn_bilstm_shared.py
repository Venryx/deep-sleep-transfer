# @sp: newly created

import keras
from keras import Input, models
from keras.layers import Conv1D, BatchNormalization, Activation, add, ZeroPadding1D, MaxPooling1D, Dropout, concatenate, \
    Bidirectional, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

import os
import sys

sys.path.append(os.getcwd())  # puts all uploaded python modules into the python path
sys.path.append('/input/src/')

import params.polyaxon_parsing_iitnet_cnn_lstm as pp3
from params.Params_Winslow import Params, winslow_params


def setup_iitnet_conv_convolutional_block(filters,
                                          kernel_size=3,
                                          strides=2):
    """
    Implementation of the identity block (or bottleneck block) as described in
    the paper to ResNet50 (https://arxiv.org/pdf/1512.03385.pdf) and implemented
    in Keras. Altered as described in Back (2019), p.5

    :param strides:
    :param filters:     numbers of filters in the 3 convolutional layers (main path)
    :param kernel_size: default 3, kernel size of the middle convolutional layer (main path)

    :return:
    """

    # get the number of filters for the three convolutional layers
    filters1, filters2, filters3 = filters

    layers_a = []
    layers_a.append(Conv1D(filters=filters3,
                           kernel_size=1,
                           strides=strides,
                           kernel_initializer='he_normal'))
    layers_a.append(BatchNormalization())

    layers_b = []
    layers_b.append(Conv1D(filters=filters1,
                           kernel_size=1,
                           strides=strides,
                           kernel_initializer='he_normal'))
    layers_b.append(BatchNormalization())
    layers_b.append(Activation('relu'))

    layers_b.append(Conv1D(filters=filters2,
                           kernel_size=kernel_size,
                           strides=1,
                           padding='same',
                           kernel_initializer='he_normal'))
    layers_b.append(BatchNormalization())
    layers_b.append(Activation('relu'))

    layers_b.append(Conv1D(filters=filters3,
                           kernel_size=1,
                           strides=1,
                           padding='valid',
                           kernel_initializer='he_normal'))
    layers_b.append(BatchNormalization())

    layers_c = []
    layers_c.append(Activation('relu'))

    conv_layers = {'layers_a': layers_a,
                   'layers_b': layers_b,
                   'layers_c': layers_c}

    return conv_layers


def setup_iitnet_conv_identity_block(filters,
                                     kernel_size=3):
    """
    Implementation of the identity block (or bottleneck block) as described in
    the paper to ResNet50 (https://arxiv.org/pdf/1512.03385.pdf) and implemented
    in Keras. Altered as described in Back (2019), p.5

    :param filters:     numbers of filters in the 3 convolutional layers (main path)
    :param kernel_size: default 3, kernel size of the middle convolutional layer (main path)

    :return:
    """

    # get the number of filters for the three convolutional layers
    filters1, filters2, filters3 = filters

    layers_b = []
    layers_b.append(Conv1D(filters=filters1,
                           kernel_size=1,
                           strides=1,
                           padding='valid',
                           kernel_initializer='he_normal'))
    layers_b.append(BatchNormalization())
    layers_b.append(Activation('relu'))

    layers_b.append(Conv1D(filters=filters2,
                           kernel_size=kernel_size,
                           strides=1,
                           padding='same',
                           kernel_initializer='he_normal'))
    layers_b.append(BatchNormalization())
    layers_b.append(Activation('relu'))

    layers_b.append(Conv1D(filters=filters3,
                           kernel_size=1,
                           strides=1,
                           padding='valid',
                           kernel_initializer='he_normal'))
    layers_b.append(BatchNormalization())

    layers_c = []
    layers_c.append(Activation('relu'))

    identity_layers = {'layers_b': layers_b,
                       'layers_c': layers_c}

    return identity_layers


def setup_first_stage_layers(stage1_filters):
    layers_start = []
    layers_start.append(ZeroPadding1D(padding=3))
    layers_start.append(Conv1D(filters=stage1_filters,
                               kernel_size=2,
                               padding='valid',
                               kernel_initializer='he_normal'))
    layers_start.append(BatchNormalization())
    layers_start.append(Activation('relu'))
    layers_start.append(ZeroPadding1D(padding=1))
    layers_start.append(MaxPooling1D(pool_size=3,
                                     strides=2))

    return layers_start


def setup_convolutional_layers(first_filters):
    # set the numbers of filters in the stages of the model
    stage1_filters = first_filters
    stage2_filters = [first_filters, first_filters, first_filters * 2]
    stage3_filters = [first_filters * 2, first_filters * 2, first_filters * 4]
    stage4_filters = [first_filters * 4, first_filters * 4, first_filters * 8]
    stage5_filters = [first_filters * 8, first_filters * 8, first_filters * 16]

    # get the layers of the first stage
    layers_stage1 = setup_first_stage_layers(stage1_filters=stage1_filters)

    # get the layers of the second stage
    layers_2_conv = setup_iitnet_conv_convolutional_block(filters=stage2_filters,
                                                          kernel_size=3,
                                                          strides=1)
    layers_2_ident_a = setup_iitnet_conv_identity_block(filters=stage2_filters,
                                                        kernel_size=3)
    layers_2_ident_b = setup_iitnet_conv_identity_block(filters=stage2_filters,
                                                        kernel_size=3)
    layers_stage2 = [layers_2_conv,
                     layers_2_ident_a,
                     layers_2_ident_b]

    # get the layers of the third stage
    layers_3_conv = setup_iitnet_conv_convolutional_block(filters=stage3_filters,
                                                          kernel_size=3)
    layers_3_ident_a = setup_iitnet_conv_identity_block(filters=stage3_filters,
                                                        kernel_size=3)
    layers_3_ident_b = setup_iitnet_conv_identity_block(filters=stage3_filters,
                                                        kernel_size=3)
    layers_3_ident_c = setup_iitnet_conv_identity_block(filters=stage3_filters,
                                                        kernel_size=3)
    layers_stage3 = [layers_3_conv,
                     layers_3_ident_a,
                     layers_3_ident_b,
                     layers_3_ident_c]

    # get the layers of the fourth stage
    layers_4_conv = setup_iitnet_conv_convolutional_block(filters=stage4_filters,
                                                          kernel_size=3)
    layers_4_ident_a = setup_iitnet_conv_identity_block(filters=stage4_filters,
                                                        kernel_size=3)
    layers_4_ident_b = setup_iitnet_conv_identity_block(filters=stage4_filters,
                                                        kernel_size=3)
    layers_4_ident_c = setup_iitnet_conv_identity_block(filters=stage4_filters,
                                                        kernel_size=3)
    layers_4_ident_d = setup_iitnet_conv_identity_block(filters=stage4_filters,
                                                        kernel_size=3)
    layers_4_ident_e = setup_iitnet_conv_identity_block(filters=stage4_filters,
                                                        kernel_size=3)
    layers_stage4 = [layers_4_conv,
                     layers_4_ident_a,
                     layers_4_ident_b,
                     layers_4_ident_c,
                     layers_4_ident_d,
                     layers_4_ident_e]

    # get the layers of the fifth stage
    layers_5_conv = setup_iitnet_conv_convolutional_block(filters=stage5_filters,
                                                          kernel_size=3)
    layers_5_ident_a = setup_iitnet_conv_identity_block(filters=stage5_filters,
                                                        kernel_size=3)
    layers_5_ident_b = setup_iitnet_conv_identity_block(filters=stage5_filters,
                                                        kernel_size=3)
    layers_stage5 = [layers_5_conv,
                     layers_5_ident_a,
                     layers_5_ident_b]

    return layers_stage1, layers_stage2, layers_stage3, layers_stage4, layers_stage5


def connect_stage_layers(layers_lists, x_input):
    output_x = 0

    for i, layer_list in enumerate(layers_lists):
        # first is the list of the conv_block
        if i == 0:
            # shortcut path
            x_short = 0
            for i, shortcut_layer in enumerate(layer_list['layers_a']):
                if i == 0:
                    x_short = shortcut_layer(x_input)
                else:
                    x_short = shortcut_layer(x_short)
            # main path
            x_main = 0
            for i, main_layer in enumerate(layer_list['layers_b']):
                if i == 0:
                    x_main = main_layer(x_input)
                else:
                    x_main = main_layer(x_main)
            # combination
            x_combined = add([x_short, x_main])
            for i, last_layer in enumerate(layer_list['layers_c']):
                x_combined = last_layer(x_combined)

            output_x = x_combined

        # rest are identity blocks
        else:
            # shortcut path
            x_short = output_x
            # main path
            x_main = 0
            for i, main_layer in enumerate(layer_list['layers_b']):
                if i == 0:
                    x_main = main_layer(output_x)
                else:
                    x_main = main_layer(x_main)
            # combination
            x_combined = add([x_short, x_main])
            for i, last_layer in enumerate(layer_list['layers_c']):
                x_combined = last_layer(x_combined)

            output_x = x_combined

        return output_x


def create_convolutional_layers(input_layers, first_filters):
    layers_stage1, layers_stage2, layers_stage3, layers_stage4, layers_stage5 = setup_convolutional_layers(
        first_filters)

    # print(layers_stage1)
    # print(layers_stage2)

    output_layers = []

    for input_layer in input_layers:
        # connect to first stage layers
        x_1 = 0
        for i, first_stage_layer in enumerate(layers_stage1):
            if i == 0:
                x_1 = first_stage_layer(input_layer)
            else:
                x_1 = first_stage_layer(x_1)

        # connect to second stage layers
        x_2 = connect_stage_layers(layers_stage2, x_1)

        # connect to third stage layers
        x_3 = connect_stage_layers(layers_stage3, x_2)

        x_3 = MaxPooling1D(pool_size=2)(x_3)

        # connect to fourth stage layers
        x_4 = connect_stage_layers(layers_stage4, x_3)

        # connect to fifth stage layers
        x_5 = connect_stage_layers(layers_stage5, x_4)

        x_5 = Dropout(0.7)(x_5)

        output_layers.append(x_5)

    return output_layers


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
        subepoch_length_points = int(subepoch_length_points * (1 + (l_overlapping / 100)))

    # define the shape of all model inputs (= shape of the subepochs)
    input_shape = (int(subepoch_length_points), n_channels)

    # make a list of input layers
    input_layers = []
    for i in range(n_subepochs_input):
        input_layer = Input(input_shape)
        input_layers.append(input_layer)

    # build the convolutional layers
    output_x = create_convolutional_layers(input_layers=input_layers,
                                           first_filters=n_filters)

    if n_subepochs_input > 1:
        x = concatenate(output_x)
    else:
        x = output_x[0]

    # RECURRENT LAYERS
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128))(x)

    # CLASSIFICATION LAYER
    x = Dense(n_classes,
              activation='softmax',
              name='fc' + str(n_classes))(x)

    # CREATE MODEL
    model = Model(list(input_layers), x, name='IITNet')

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


# see https://stackoverflow.com/questions/48198031/how-to-add-variables-to-progress-bar-in-keras/48206009#48206009
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


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

    optimizer = Adam(lr=learning_rate,
                     beta_1=beta1,
                     beta_2=beta2)

    # see link above
    # optimizer = SGD(lr=learning_rate)
    # lr_metric = get_lr_metric(optimizer)

    # model_gpu = keras.utils.multi_gpu_model(model, gpus=2)
    # model_gpu.compile(loss='categorical_crossentropy',
    #                   optimizer=optimizer,
    #                   metrics=['accuracy'])
    # return model_gpu

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def create_model_iitnet(params, model, do_compile=True):
    """
    Build and compile an IITNet model, for more details see paper from Back (2019)
    :param do_compile: whether to compile the model before returning
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

        if do_compile:
            print("\nCompiling model ... ", end=" ")
            model = compile_model_iitnet(params, model)
            print("done")

        model.summary()

    return model

# if __name__ == "__main__":
#     # get parameters
#     params = Params()
#
#     # get additional parameters for iitnet
#     plx: dict = pp3.get_parameters()
#     params.plx.update(plx)
#     # params.plx['batch_size'] = 250
#     params.plx['subject_batch'] = 1  # !
#     params.plx['apply_downsampling'] = True     # param common_frequency has to be set
#     # NOTE: mdl_architecture has to be set to 'iitnet_cnn_bilstm'
#
#     # adjust winslow parameters
#     if 'WINSLOW_PIPELINE_NAME' in os.environ:
#         winslow_params(params)
#
#     # Build model
#     model = create_model_iitnet(params, model=None, compile=True)
