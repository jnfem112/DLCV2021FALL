from p2_utils import my_argparse , set_random_seed
import p2_model

def main(args):
	set_random_seed(0)
	model = getattr(p2_model , args.model)(args.input_dim , args.number_of_class)
	model.train(args)

if __name__ == '__main__':
	args = my_argparse()
	main(args)