class Config(object):
  def __init__(self):
    self.data = oppo()
    self.evaluation = evaluation()

    self.gpu = 0
    self.gpu_ids = [0, 1, 2, 3]
    self.multi_gpu = False
    self.num_split = 5
    self.augmentation = True
    self.result_root = './experiment_log/downstream/'+self.data.dataset_name+'_'+self.evaluation.evaluation_name
    self.model_path = self.result_root+'/best.mdl'
    self.logging_path = self.result_root+'/output.log'
    self.is_verbose = False

class oppo(object):
  def __init__(self):
    # self.data_root = './data/downstream/oppo_dataset'
    self.data_root = '/home/ukita/har/ssl-wearables-main/data/downstream/oppo_dataset'
    self.x_train_path = self.data_root+'/oppo_30hz_w10_o5_x_train.npy'
    self.y_train_path = self.data_root+'/oppo_30hz_w10_o5_y_train.npy'
    self.x_valid_path = self.data_root+'/oppo_30hz_w10_o5_x_val.npy'
    self.y_valid_path = self.data_root+'/oppo_30hz_w10_o5_y_val.npy'
    self.x_test_path = self.data_root+'/oppo_30hz_w10_o5_x_test.npy'
    self.y_test_path = self.data_root+'/oppo_30hz_w10_o5_y_test.npy'
    # self.PID_path = self.data_root+'/pid.npy'
    self.sample_rate = 33
    self.task_type = 'classify'
    self.output_size = 4
    self.batch_size = 100
    self.held_one_subject_out = True
    self.weighted_loss_fn = True
    self.dataset_name = 'oppo'
    self.subject_count = -1
    self.ratio2keep = 1

class evaluation(object):
  def __init__(self):
    self.model_path = './experiment_log/pre-train/task4/best.mdl'
    self.load_weights = True
    self.freeze_weight = True # true: transfer, false: fine-tune
    self.input_size = 300
    self.patch_size = 30
    self.subR = 1
    self.learning_rate = 0.0001
    self.num_workers = 6
    self.patience = 16
    self.num_epoch = 10
    self.evaluation_name = 'transfer' if self.freeze_weight else 'fine-tune'

class model(object):
  def __init__(self):
      self.learning_rate = 0.0001
      self.name = 'Transformer'
      self.mixed_precision = False
      self.warm_up_step = 5
      self.lr_scale = True
      self.patience = 10

class dataloader(object):
  def __init__(self):
    self.num_sample_per_subject = 1500
    self.sample_rate = 30
    self.epoch_len = 10