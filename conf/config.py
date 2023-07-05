class Config(object):
  def __init__(self):
    self.runtime = runtime()
    self.data = data()
    self.model = model()
    self.task = task()
    self.augmentation = augmentation()
    self.dataloader = dataloader()

class runtime(object):
  def __init__(self):
    self.gpu = 0
    self.gpu_ids = [0, 1, 2, 3]
    self.distributed = False
    self.multi_gpu = False
    self.num_epoch = 500
    self.is_epoch_data = True

class data(object):
  def __init__(self):
    self.data_root = '/home/SHL/ssl-data/ssl/ssl_capture_24'
    self.batch_subject_num = 1
    self.train_data_root = self.data_root+'/data/train'
    self.test_data_root = self.data_root+'/data/test'
    self.train_file_list = './data/ssl/ssl_capture_24/data/train/file_list.csv'
    self.test_file_list = './data/ssl/ssl_capture_24/data/test/file_list.csv'
    self.log_path = './experiment_log/pre-train'
    self.log_interval = 60
    self.data_name = 'sslCapture24'
    self.weighted_sample = True
    self.ratio2keep = 1
    self.capture24_x_path = '/home/SHL/ssl-data/downstream/capture24_30hz_full/X.npy'
    self.capture24_y_path = '/home/SHL/ssl-data/downstream/capture24_30hz_full/Y.npy'
    self.capture24_pid_path = '/home/SHL/ssl-data/downstream/capture24_30hz_full/pid.npy'

class model(object):
  def __init__(self):
    self.learning_rate = 0.0001
    self.patch_size = 30
    self.window_size = 300
    self.name = 'transformer'
    self.mixed_precision = False
    self.warm_up_step = 5
    self.lr_scale = True
    self.patience = 50

class task(object):
  def __init__(self):
    self.mask = True
    self.permute = True
    self.timewarp = True
    self.rotation = True
    self.wdba = False
    self.rgw = False
    self.dgw = False
    self.num = 4
    self.name = 'task'+str(self.num)

class augmentation(object):
  def __init__(self):
    self.axis_switch = False
    self.rotation = False

class dataloader(object):
  def __init__(self):
    self.num_sample_per_subject = 1500
    self.sample_rate = 30
    self.epoch_len = 10