from p3_utils import my_argparse
import p3_model

def main(args):
	model = getattr(p3_model , args.model)(channel = 3 if args.source != 'usps' else 1)
	model.train(args)

if __name__ == '__main__':
	args = my_argparse()
	main(args)