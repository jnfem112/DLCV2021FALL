from p1_utils import my_argparse , set_random_seed
import p1_model

def main(args):
	set_random_seed(0)
	model = getattr(p1_model , args.model)(args.input_dim)
	model.train(args)

if __name__ == '__main__':
	args = my_argparse()
	main(args)