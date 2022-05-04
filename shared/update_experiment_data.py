import argparse
import os 



if __name__ == "__main__":

	parser = argparse.ArgumentParser()


	parser.add_argument('--source', '-source',
		help="Source machine",
		default = 'MIT', 
		type=str)

	parser.add_argument('--target', '-target',
		help="Target machine",
		default ='GL', 
		type=str)

	parser.add_argument('--data_dir', '-data_dir',
		help="directory that has the experiment data",
		type=str)

	args = vars(parser.parse_args())

	if args['source'] == "MIT":
		# source_string1 = '/data/ddmg/scate/dr/images_processed'
		# source_string2 = '/data/ddmg/scate/multiple_shortcut/dr//experiment_data/'
		source_string1 = '/data/ddmg/slabs/CUB_200_2011/'
		source_string2 = '/data/ddmg/slabs/waterbirds/places_data/'

	if args['target'] == "GL":
		# target_string1 = '/nfs/turbo/coe-soto/dr/images_processed'
		# target_string2 =  '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/dr/experiment_data/'
		target_string1 = '/nfs/turbo/coe-soto/CUB_200_2011/'
		target_string2 = '/nfs/turbo/coe-soto/places_data/'

	txtfiles = []
	for i in range(10):
		rspath = f'{args["data_dir"]}/rs{i}'
		txtfiles = txtfiles + [f for f in os.listdir(rspath) if f[-4:] == '.txt']
		for filename in txtfiles:
			print(f'{args["data_dir"]}/rs{i}/{filename}')
			# Read in the file
			with open(f'{args["data_dir"]}/rs{i}/{filename}', 'r') as file :
				filedata = file.read()

			# Replace the target string
			filedata = filedata.replace(source_string1, target_string1)
			filedata = filedata.replace(source_string2, target_string2)

			# Write the file out again
			with open(f'{args["data_dir"]}/rs{i}/{filename}', 'w') as file:
				file.write(filedata)