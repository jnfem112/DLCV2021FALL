import math
import torch
import matplotlib.pyplot as plt
from p1_model import myViT

def cosine_similarity(x1 , x2):
	return torch.matmul(x1 , x2.T) / (torch.norm(x1 , dim = 1).view(-1 , 1) + 1e-8) / (torch.norm(x2 , dim = 1) + 1e-8)

def visualize_positional_embedding(model):
	positional_embedding = model.positional_embedding.pos_embedding.squeeze()[1 : ]
	result = cosine_similarity(positional_embedding , positional_embedding).detach().numpy()
	width = int(math.sqrt(result.shape[0]))
	fig = plt.figure(figsize = (20 , 20))
	for i in range(result.shape[0]):
		ax = fig.add_subplot(width , width , i + 1)
		ax.imshow(result[i].reshape(width , width) , cmap = 'viridis')
		ax.set_axis_off()
	plt.savefig('positional_embedding.png')

def main():
	model = myViT()
	model.load_state_dict(torch.load('ViT.pth'))
	visualize_positional_embedding(model)

if __name__ == '__main__':
	main()