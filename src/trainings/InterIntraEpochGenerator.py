# @sp: newly created

import numpy as np
from math import ceil

from imblearn.over_sampling import RandomOverSampler, SMOTE
from keras.utils import Sequence
from scipy.signal import resample


class InterIntraEpochGenerator(Sequence):

    def __init__(self, data_int, params, num_subjects, start_val=0,
                 evaluation=False, shuffle=False, oversampling=False, crossval_samples=0):
        """

        :param data_int:
        :param params:
        :param num_subjects: number of subjects used for this step (train/validation/test)
        :param start_val:    subject index to start from when getting subject data
        :param evaluation:   boolean, whether this generator is used in evaluation step
        :param shuffle:      boolean, whether to shuffle the data before outputting batches
        """

        self.data_int = data_int
        self.params = params
        self.start_val = start_val
        self.total_num_subjects = num_subjects
        self.shuffle = shuffle
        self.oversampling = oversampling

        self.steps_per_epoch = None

        self.evaluation = evaluation
        self.label_list = []

        self.num_loaded_subjects = 0
        self.number_of_prev_batches = 0
        self.batches_x = []
        self.batches_y = []
        self.leftover_x_data = np.array([])
        self.leftover_y_data = np.array([])

        self.l_epochs = 0
        self.l_subepochs = 0
        self.l_overlap = 0
        self.crossval_samples = crossval_samples

        self.class_weights = 0

        self.setup()

    def __getitem__(self, item):
        """
        returns a batch of subsamples
        :param item: index of the batch
        """
        batch = self.check_item(item)
        self.batches_y[batch] = np.asarray([x for x in self.batches_y[batch]])
        return self.batches_x[batch], self.batches_y[batch]

    def __len__(self):
        return self.steps_per_epoch

    def setup(self):
        """
        Setup the variables for the Generator-object
        """
        self.l_epochs = self.params.plx.get('l_epochs')
        self.l_subepochs = self.params.plx.get('l_subepochs')
        self.l_overlap = self.params.plx.get('l_overlapping')

        if self.oversampling:
            labels, class_distributions = calc_class_distributions_oversampled(self.data_int,
                                                                               self.start_val,
                                                                               self.total_num_subjects,
                                                                               self.params)
        else:
            labels, class_distributions = calc_class_distributions(self.data_int,
                                                                   self.start_val,
                                                                   self.total_num_subjects,
                                                                   self.params)

        # get the class weights
        largest_class = np.amax(class_distributions)
        class_weights = class_distributions / largest_class
        fit_weights = 1 / class_weights
        weights_dict = {i: fit_weights[i] for i in range(0, len(fit_weights))}

        self.class_weights = weights_dict

        self.calc_steps_per_epoch(class_distributions)

    def calc_steps_per_epoch(self, class_distributions):
        """
        Calculates the number of steps (batches of subsamples) per epoch and saves it to the
        internal variable of the object.
        :return: int; ceil(total_num_samples / batch_size)
        """
        total_subjects = self.total_num_subjects
        total_labels = np.sum(class_distributions)
        total_subsamples = total_labels - (total_subjects * (self.l_epochs - 1))

        self.steps_per_epoch = ceil(total_subsamples / self.params.plx["batch_size"])

    def check_item(self, item):
        """
        Checks, if there are enough batches left in the objects stash or if a new subject
        has to be loaded and processed into batches. Returns the batch_index relative to the
        loaded data in the objects stash = relative to the loaded subject batch.
        :param item: index of the batch to be loaded, coming from the keras fit_generator
        :return relative_item: batch_index relative to the current subject batch
        """
        if item == 0:
            # if this is a completely new epoch = completely new data load
            self.reset_counters()
            self.load_next_subject_batch()
        elif item - self.number_of_prev_batches > len(self.batches_x) - 1:
            # if this is the last batch in the backlog
            self.load_next_subject_batch()
            self.number_of_prev_batches = item
        relative_item = item - self.number_of_prev_batches
        return relative_item

    def load_next_subject_batch(self):
        """
        If more batches are needed, load a new subject, turn it into subsamples and make
        batches of the subsamples. If there are subsamples left from the previous subject,
        add them to this batch. Save those batches to the backlog.
        """
        if not self._all_subjects_loaded():
            # if there are still subjects left that can be loaded
            data_x, data_y = self.load_and_process_data()
            self.make_x_y_batches(data_x, data_y)
        else:
            # if all subjects are already loaded, make the last batch with the leftover subsamples
            self.batches_x = [subbatch_to_modelinput(self.leftover_x_data, self.params)]
            self.batches_y = [self.leftover_y_data]
        # update the global list of all labels, used for evaluation
        if self.evaluation and len(self.label_list) < len(self):
            self.batches_y = np.asarray([np.asarray([y for y in x]) for x in self.batches_y])
            self.label_list.extend(self.batches_y)

    def load_and_process_data(self):
        """
        Load the x and y data from the subject. Process the data into subsamples, each consisting of
        L epochs with l subepochs each. Label of the subsample = label of last epoch.
        Append the stored leftover subsamples from last subject and return the array of subsamples.
        """
        data_x, data_y = self.load_new_subject()

        data_x = data_x[:, 0, :]  # shape (x, 1, 3000) to (x, 3000)

        if self.params.plx.get('apply_downsampling'):
            data_x = self.downsample_frequency(data_x)

        subsamples = self.data_to_subsamples(data_x, data_y)
        subsamples = np.asarray(subsamples)
        subsamples_x = subsamples[:, 0]
        subsamples_y = subsamples[:, 1]

        subsamples_x, subsamples_y = self._append_leftovers_to(subsamples_x, subsamples_y)
        # Oversample the subsamples and classes
        if self.oversampling:
            subsamples_x, subsamples_y = self.oversample_subject_data(subsamples_x, subsamples_y)
        # shuffle the subsamples in this subject
        if self.shuffle:
            subsamples_x, subsamples_y = shuffle_subject_data(subsamples_x, subsamples_y)

        return subsamples_x, subsamples_y

    def data_to_subsamples(self, data_x, data_y):
        """
        Part a subject with X epochs into subsamples of the length L epochs, part each epoch into l subepochs
        :param data_x:
        :param data_y:
        """

        l_epochs = self.l_epochs
        l_subepochs = self.l_subepochs
        if self.params.plx.get('apply_downsampling'):
            frequency = self.params.plx.get('common_frequency')
        else:
            frequency = self.params.plx.get('frequency')

        subsamples = []

        n_epochs_subject = data_x.shape[0]
        n_subsamples = n_epochs_subject - (l_epochs - 1)

        for subsample_index in range(n_subsamples):
            subsample_x = data_x[subsample_index: (l_epochs + subsample_index)]

            # split every epoch into l subepochs
            subsample_x_split = np.asarray(self.subsample_to_subepochs(subsample_x, l_subepochs, frequency))
            # label of each subsample is label of the last epoch in the subsample
            subsample_y = data_y[l_epochs + (subsample_index - 1)]

            subsample = np.asarray([subsample_x_split, subsample_y])
            subsamples.append(subsample)

        return subsamples

    def subsample_to_subepochs(self, subsample_x, l_subepochs, frequency):
        """
        Split every epoch in the subsample into l overlapping subepochs
        :param subsample_x:
        :param l_subepochs:
        :param frequency:
        """

        epoch_length_sec = self.params.plx.get('sections')
        l_overlap = self.l_overlap

        datapoints_per_epoch = epoch_length_sec * frequency
        datapoints_per_subepoch = datapoints_per_epoch / l_subepochs

        # subepochs overlap, so they have to be longer by l percent
        subepoch_length_with_overlap = int(datapoints_per_subepoch * (1 + (l_overlap / 100)))
        overlap_in_datapoints = int(datapoints_per_subepoch * (l_overlap / 100))

        subepochs = []

        for idx_epoch, epoch in enumerate(subsample_x):
            for idx_subepoch in range(l_subepochs):
                # all subepochs have to have the same length, so the overlap between first and second
                # and between last and second to last is bigger than between the others (there are methods
                # to avoid this, but for now, keep it simple)
                if idx_subepoch == 0:
                    # if this is the first subepoch in the epoch
                    start_index = 0
                    end_index = subepoch_length_with_overlap
                elif idx_subepoch == l_subepochs - 1:
                    # if this is the last subepoch in the epoch
                    start_index = datapoints_per_epoch - subepoch_length_with_overlap
                    end_index = datapoints_per_epoch
                else:
                    start_index = int((datapoints_per_subepoch * idx_subepoch) - (overlap_in_datapoints / 2))
                    end_index = int(datapoints_per_subepoch * (idx_subepoch + 1) + (overlap_in_datapoints / 2))

                subepoch = epoch[start_index:end_index]

                subepochs.append(subepoch)

        return subepochs

    def make_x_y_batches(self, data_x, data_y):
        """
        Make the input data (subsamples) into batches of the specified size. These will then be used
        to train the model. If there are subsamples left, store them and use them for the next subject batch.
        :param data_x: subsamples_x
        :param data_y: subsamples_y
        """
        self.batches_x, self.leftover_x_data = make_batches_x(data_x, self.params, self.params.plx["batch_size"])
        self.batches_y, self.leftover_y_data = make_batches_y(data_y, self.params.plx["batch_size"])

    def load_new_subject(self):
        """
        Load new subject via the DataInterface. Keep count, how many subjects have been loaded
        so far.
        """
        # if the index of the subject to be loaded is larger than the training dataset,
        # start again at the index 0
        if self.crossval_samples is not 0:
            if self.num_loaded_subjects + self.start_val >= self.crossval_samples:
                self.start_val = 0 - self.num_loaded_subjects

        data_x, data_y = self.data_int.load_particular_data(
            uuid=self.params.plx["experiment_uuid"],
            num_subjects=1,
            process_step_name=self.params.plx["feature_eng_filename"],
            offset=self.num_loaded_subjects + self.start_val,
            params=self.params)
        self.num_loaded_subjects += 1

        return data_x, data_y

    def reset_counters(self):
        self.num_loaded_subjects = 0
        self.number_of_prev_batches = 0
        self.leftover_x_data = np.array([])
        self.leftover_y_data = np.array([])

    def _all_subjects_loaded(self):
        """
        Check if all subjects have been loaded yet
        """
        if self.total_num_subjects == self.num_loaded_subjects:
            return True
        return False

    def _append_leftovers_to(self, data_x, data_y):
        if self.leftover_x_data.size > 0:
            data_x = np.asarray([x for x in data_x])
            data_y = np.asarray([x for x in data_y])
            self.leftover_x_data = np.asarray([x for x in self.leftover_x_data])
            self.leftover_y_data = np.asarray([x for x in self.leftover_y_data])
            data_x = np.append(data_x, self.leftover_x_data, axis=0)
            data_y = np.append(data_y, self.leftover_y_data, axis=0)
        return data_x, data_y

    def downsample_frequency(self, data_x):
        """
        Downsample the frequency of the signal data to the common frequency
        :param data_x:
        :param data_y:
        """
        target_frequency = self.params.plx.get('common_frequency')
        epoch_length_sec = self.params.plx.get('sections')
        target_length = target_frequency * epoch_length_sec

        data_x_target = []

        for epoch in data_x:
            data_x_epoch = resample(epoch, target_length)

            data_x_target.append(data_x_epoch)

        return np.array(data_x_target)

    def oversample_subject_data(self, subsamples_x, subsamples_y):
        """
        Oversample the classes, so that there are an equal number of each class in the subject
        Very simple oversampling: duplicate datapoints in the minority classes
        :param subsamples_x:
        :param subsamples_y:
        """

        # ros = SMOTE(random_state=0)
        ros = RandomOverSampler(sampling_strategy='not majority', random_state=0)

        y_as_array = np.asarray([x for x in subsamples_y])
        x_as_array = np.asarray([x for x in subsamples_x])
        x_2d = np.asarray([x.reshape((x_as_array.shape[1] * x_as_array.shape[2])) for x in x_as_array])

        # bugfix: check if there is one class with no values in y_as_array
        missing_classes = []
        for label in range(5):
            all_labels_this_class = y_as_array[:, label]
            if 1 not in all_labels_this_class:
                missing_classes.append(label)

        x_resampled, y_resampled = ros.fit_resample(x_2d, y_as_array)

        x_3d = np.asarray([x.reshape((x_as_array.shape[1], x_as_array.shape[2])) for x in x_resampled])

        # bugfix: add the missing values again, they are removed by the Oversampler
        if len(missing_classes) > 0:
            y_repaired = []
            for element in y_resampled:
                for missing in missing_classes:
                    new_element = element.tolist()
                    new_element.insert(missing, 0)
                    new_element = np.asarray(new_element)
                y_repaired.append(new_element)
            y_resampled = y_repaired

        y_resampled = np.asarray([x for x in y_resampled])

        return x_3d, y_resampled


def shuffle_subject_data(subsamples_x, subsamples_y):
    """
    Shuffle the subsamples in the subject
    :param subsamples_x:
    :param subsamples_y:
    """
    num_subsamples = subsamples_x.shape[0]
    shuffled_indices = np.random.permutation(np.arange(num_subsamples))

    subsamples_x = subsamples_x[shuffled_indices]
    subsamples_y = subsamples_y[shuffled_indices]

    return subsamples_x, subsamples_y


def make_batches_y(data_y, batch_size):
    """
    Make a list of label batches from the input array
    :param data_y:
    :param batch_size:
    """
    num_batches = data_y.shape[0] // batch_size
    # trim data so that every batch is the size batch_size
    trimmed_data = data_y[:num_batches * batch_size]
    batches = np.split(trimmed_data, num_batches)
    leftover_subsamples = data_y[num_batches * batch_size:]

    return batches, leftover_subsamples


def make_batches_x(data_x, params, batch_size):
    """
    Turn the list of input subsamples into batches
    :param data_x:
    :param batch_size:
    """
    num_full_batches = data_x.shape[0] // batch_size
    full_batches = []

    batch_start = 0
    for batch_idx in range(num_full_batches):
        batch_start = batch_idx * batch_size    # fix 12.06.2020
        batch_end = batch_start + batch_size
        batch_x = data_x[batch_start:batch_end]

        # Transform into model input
        model_input_x = subbatch_to_modelinput(batch_x, params)

        full_batches.append(model_input_x)

    leftover_subsamples = data_x[num_full_batches * batch_size:]

    return full_batches, leftover_subsamples


def subbatch_to_modelinput(batch_x, params):
    """
    Reformat the array of subsamples into an array of L*l "strands", each containing one subepoch
    :param batch_x: array of batch_size subepochs
    :param params:
    """
    subepoch_strands = []

    n_subepochs_per_subbatch = params.plx.get('l_epochs') * params.plx.get('l_subepochs')
    n_channels = len(params.plx.get("ch_idx_list"))
    for subepoch_index in range(n_subepochs_per_subbatch):
        strand = []

        for subbatch_index in range(len(batch_x)):
            strand.append(batch_x[subbatch_index][subepoch_index])

        strand = np.asarray(strand)

        n_subbatches = strand.shape[0]
        datapoints_subepoch = strand.shape[1]

        strand = strand.reshape((n_subbatches, datapoints_subepoch, n_channels))

        subepoch_strands.append(strand)

    return subepoch_strands


def calc_class_distributions(data_int, start_val, num_subjects, params=None):
    """
    :return:
    """
    distrib_dict = data_int.get_distribution_matrix(start_val)
    labels = list(distrib_dict.keys())
    class_distributions = np.array(list(distrib_dict.values()), dtype=int)
    if params is not None:
        total_subjects = params.plx.get('train_count') + params.plx.get('val_count')

    for subject in range(1, num_subjects):
        subject = subject + start_val

        if total_subjects is not None:
            if subject >= total_subjects:
                subject = subject - total_subjects

        subj_dict = data_int.get_distribution_matrix(subject)
        subj_distributions = np.array(list(subj_dict.values()), dtype=int)
        class_distributions = class_distributions + subj_distributions

    return labels, class_distributions


def calc_class_distributions_oversampled(data_int, start_val, num_subjects, params=None):
    """
    calculate class distributions if the subjects are oversampled
    """

    # get the class distribution for each subject as array
    all_distrib_list = []

    if params is not None:
        total_subjects = params.plx.get('train_count') + params.plx.get('val_count')

    for subject in range(num_subjects):
        subject = subject + start_val

        if total_subjects is not None:
            if subject >= total_subjects:
                subject = subject - total_subjects

        subj_dict = data_int.get_distribution_matrix(subject)
        subj_distributions = np.array(list(subj_dict.values()), dtype=int)

        # get the labels (only in the first iteration)
        if subject == start_val:
            labels = list(subj_dict.keys())

        # oversample the distribution
        max_class = np.amax(subj_distributions)         # highest class in the sample

        oversampled_distribution = []
        for label in subj_distributions:
            if not label == 0:
                oversampled_distribution.append(max_class)
            else:
                oversampled_distribution.append(0)
        oversampled_distribution = np.asarray(oversampled_distribution)

        all_distrib_list.append(oversampled_distribution)

    # add the distributions together
    all_distributions = all_distrib_list[0]

    if len(all_distrib_list) > 1:
        for idx in range(1, len(all_distrib_list)):
            all_distributions = all_distributions + all_distrib_list[idx]

    return labels, all_distributions
