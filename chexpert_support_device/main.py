"""Main file for running the chexpert experiment."""

from absl import app
from absl import flags

from chexpert_support_device import data_builder
from shared import train
from shared import train_fs
from shared.train_utils import restrict_GPU_tf

FLAGS = flags.FLAGS
flags.DEFINE_string('skew_train', 'False', 'train on skewed data?')
flags.DEFINE_enum('v_mode', 'normal', 
	['normal', 'noisy', 'corry'],
	'create additional V dimensions? of which kind?')

flags.DEFINE_float('p_tr', .7, 'proportion of data used for training.')
flags.DEFINE_float('p_val', .25,
	'proportion of training data used for validation.')
flags.DEFINE_integer('pixel', 128,
	'number of pixels in the image (i.e., resolution).')
flags.DEFINE_string('data_dir', '/datadir/',
										'Directory of saved data.')
flags.DEFINE_string('exp_dir', '/mydir/',
										'Directory to save trained model in.')
flags.DEFINE_string('checkpoint_dir', '/mycheckpointdir/',
										'Directory to save the checkpoints of the model in.')
flags.DEFINE_string('architecture', 'pretrained_densenet',
										'Architecture to use for training.')
flags.DEFINE_integer('batch_size', 64, 'batch size.')
flags.DEFINE_integer('num_epochs', 100, 'Number of epochs.')
flags.DEFINE_integer('training_steps', 0, 'number of estimator training steps.'
										' If non-zero over rides the automatic value'
										' determined by num_epochs and batch_size')
flags.DEFINE_float('alpha', 1.0, 'Value for the cross prediction penelty')
flags.DEFINE_float('sigma', 1.0, 'Value for the MMD kernel bandwidth.')
flags.DEFINE_string('weighted', 'False', 'use weighting?.')
flags.DEFINE_string('conditional_hsic', 'False',
											'compute HSIC conditional on y?.')
flags.DEFINE_float('l2_penalty', 0.0,
									'L2 regularization penalty for final layer')
flags.DEFINE_integer('embedding_dim', -1,
										'Dimension for the final embedding.')
flags.DEFINE_integer('random_seed', 0, 'random seed for tensorflow estimator')
flags.DEFINE_integer('v_dim', 0, 'dimension of additional aux labels')
flags.DEFINE_enum('alg_step', 'None', ['None', 'first', 'second'],
	'Are you running the two step alg? If so, which step?')
flags.DEFINE_string('cleanup', 'False',
		'remove tensorflow artifacts after training to reduce memory usage.')
flags.DEFINE_string('gpuid', '0', 'Gpu id to run the model on.')
flags.DEFINE_string('warm_start_dir', 'None', 'directory to warmstart from.')
flags.DEFINE_string('debugger', 'False', 'debugger mode')


def main(argv):

	del argv

	def dataset_builder():
		return data_builder.build_input_fns(
			chexpert_data_dir=FLAGS.data_dir,
			v_mode=FLAGS.v_mode,
			skew_train=FLAGS.skew_train,
			weighted=FLAGS.weighted,
			p_tr=FLAGS.p_tr,
			p_val=FLAGS.p_val,
			v_dim=FLAGS.v_dim, 
			random_seed=FLAGS.random_seed, 
			alg_step=FLAGS.alg_step)

	restrict_GPU_tf(FLAGS.gpuid, memfrac=0.9)

	py1_y0_shift_list = [0.1, 0.5, 0.9]

	if FLAGS.alg_step != "first":
		train.train(
			exp_dir=FLAGS.exp_dir,
			checkpoint_dir=FLAGS.checkpoint_dir,
			dataset_builder=dataset_builder,
			architecture=FLAGS.architecture,
			training_steps=FLAGS.training_steps,
			pixel=FLAGS.pixel,
			n_classes=1, 
			num_epochs=FLAGS.num_epochs,
			batch_size=FLAGS.batch_size,
			alpha=FLAGS.alpha,
			sigma=FLAGS.sigma,
			weighted=FLAGS.weighted,
			conditional_hsic=FLAGS.conditional_hsic,
			l2_penalty=FLAGS.l2_penalty,
			embedding_dim=FLAGS.embedding_dim,
			random_seed=FLAGS.random_seed,
			cleanup=FLAGS.cleanup,
			py1_y0_shift_list=py1_y0_shift_list,
			warm_start_dir=FLAGS.warm_start_dir,
			debugger=FLAGS.debugger)
	else: 
		train_fs.train(
			exp_dir=FLAGS.exp_dir,
			checkpoint_dir=FLAGS.checkpoint_dir,
			dataset_builder=dataset_builder,
			architecture=FLAGS.architecture,
			training_steps=FLAGS.training_steps,
			pixel=FLAGS.pixel,
			n_classes=FLAGS.v_dim,
			num_epochs=FLAGS.num_epochs,
			batch_size=FLAGS.batch_size,
			weighted=FLAGS.weighted,
			l2_penalty=FLAGS.l2_penalty,
			embedding_dim=FLAGS.embedding_dim,
			random_seed=FLAGS.random_seed,
			cleanup=FLAGS.cleanup,
			debugger=FLAGS.debugger)

if __name__ == '__main__':
	app.run(main)
