import tensorflow as tf

class readCSV():
    def __init__(self, n_inputs, mean, std, **kwargs): # n_inputs = features X
        super().__init__(**kwargs)
        self.n_inputs = n_inputs 
        self.mean = mean
        self.std = std   
        
    # data transformation
    @tf.function
    def preprocess(self, line): # line = 8 features, 1 outcome --> n_inputs = 8
        # default : features = [0., 0., ...], outcome = [] 
        defs = [0.] * self.n_inputs + [tf.constant([], dtype=tf.float32)] 
        # decode csv 
        fields = tf.io.decode_csv(line, record_defaults=defs)
        # split features & label
        x = tf.stack(fields[:-1]) # stack for features X
        y = tf.stack(fields[-1:]) # stack for outcome
        # normalization
        self.z = (x - self.mean) / self.std
        return self.z, y 

    # read dataset by DataAPI
    def csv_reader_dataset(self, filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
                       
        # input pipeline
        dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
        dataset = dataset.interleave(
            lambda filepath: tf.data.TextLineDataset(filepath).skip(1), #skip head
            cycle_length=n_readers, num_parallel_calls=n_read_threads)
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.map(self.preprocess, num_parallel_calls=n_parse_threads)
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(1)