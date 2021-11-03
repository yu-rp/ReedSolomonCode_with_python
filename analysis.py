import matplotlib
matplotlib.use("agg")
import os,datetime,argparse,torch,numpy,pandas,seaborn,logging,scipy
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,Subset
from modules import *
from itertools import product
from matplotlib import pyplot
from scipy.stats import kstest
from utils import dec2
logging.basicConfig(filename="../results/log.log",level = logging.INFO)
logger = logging.getLogger(__name__)

print(os.getcwd())

def timestamp(sec = True):
    return datetime.datetime.strftime(datetime.datetime.now(),r"%m%d_%H%M%S%f") if sec else datetime.datetime.strftime(datetime.datetime.now(),r"%m%d_%H%M")

seed = 0

os.environ['PYTHONHASHSEED'] = str(seed) if "os" in globals() else None
numpy.random.seed(seed) if "numpy" in globals() else None
random.seed(seed) if "random" in globals() else None
if "torch" in globals():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

class AnalysisZ:
    def __init__(self, z, savefile):
        self.z = pandas.DataFrame(z)
        self.savefile = f"{savefile}"
        self.analyze()
    
    def analyze(self):
        pyplot.figure()
        fig_pair = seaborn.pairplot(self.z,diag_kind="kde")
        fig_pair.savefig(self.savefile+".png")
        pyplot.close("all")
        logger.info(f"fig save to {self.savefile}.png")
        pyplot.figure()
        fig_box = seaborn.boxplot(data = self.z)
        fig_box.get_figure().savefig(self.savefile+"_box.png")
        pyplot.close("all")
        logger.info(f"fig save to {self.savefile}_box.png")
        logger.info(f"{self.savefile}")
        for c in range(self.z.shape[1]):
            data = self.z[c].to_numpy()
            mean,var = numpy.mean(data),numpy.var(data)
            data = (data-mean)/var
            logger.info(f'{c},mean,{mean},var,{var},norm,{kstest(data,lambda x: scipy.stats.norm.cdf(x, loc=0, scale=1))}')

class AnalysisMSE:
    def __init__(self, mse, savefile):
        self.mse = pandas.DataFrame(mse)
        self.savefile = f"{savefile}"
        self.analyze()
    
    def analyze(self):
        pyplot.figure()
        fig_pair = seaborn.pairplot(self.mse,diag_kind="hist")
        fig_pair.savefig(self.savefile+".png")
        pyplot.close("all")
        logger.info(f"fig save to {self.savefile}.png")
        # pyplot.figure()
        # fig_box = seaborn.boxplot(data = self.mse)
        # fig_box.get_figure().savefig(self.savefile+"_box.png")
        # pyplot.close("all")
        # logger.info(f"fig save to {self.savefile}_box.png")
        # logger.info(f"{self.savefile}")
        # for c in range(self.mse.shape[1]):
        #     data = self.mse[c].to_numpy()
        #     mean,var = numpy.mean(data),numpy.var(data)
        #     logger.info(f"{c},mean,{mean},var,{var},norm,{kstest(data,lambda x: scipy.stats.norm.cdf((x-mean)/var))}")

def main(args):

    pixels = list(product(range(2,28,6),range(2,28,6)))

    logger.info('args:')
    for k,v in sorted(vars(args).items()):
        logger.info(f"\t{k}: {v}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MNIST(root='../../data', train=True, transform=transforms.ToTensor(),download=True)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

    subset = list(DataLoader(Subset(dataset,range(10)),batch_size=10))[0][0]

    if args.model == "AE":
        net = AE(latent_dim=args.latent_dim).to(device)
        net.load_state_dict(torch.load(args.net_file,map_location=torch.device('cpu')))
        net.eval()

        zs = torch.ones(1,args.latent_dim)
        mses = torch.ones(1,28,28,8)
        for iteration, (x, _) in enumerate(data_loader):
            print(timestamp(),iteration,"/",len(data_loader)-1,end="\r")

            x = x.to(device)

            recon_x, z = net(x)
            
            x = x.detach().cpu().squeeze(1)
            recon_x = recon_x.detach().cpu().squeeze(1)
            z = z.detach().cpu()

            zs = torch.cat((zs,z),0)

            x = torch.round(x*255)
            recon_x = torch.round(recon_x*255)

            xshape = x.shape
            xcode = torch.zeros((*xshape,8))
            for n in range(xshape[0]):
                for r,c in pixels:
                    xcode[n,r,c] = torch.from_numpy(dec2(x[n,r,c]))

            recon_xcode = torch.zeros((*xshape,8))
            for n in range(xshape[0]):
                for r,c in pixels:
                    recon_xcode[n,r,c] = torch.from_numpy(dec2(recon_x[n,r,c]))

            mses = torch.cat((mses,xcode-recon_xcode),0)
        
        subset_recon,_ = net(subset)

    elif args.model == "VAE":
        net = VAE(latent_dim=args.latent_dim).to(device)
        net.load_state_dict(torch.load(args.net_file,map_location=torch.device('cpu')))
        net.eval()

        zs = torch.ones(1,args.latent_dim)
        means = torch.ones(1,args.latent_dim)
        mses = torch.ones(1,28,28,8)
        for iteration, (x, _) in enumerate(data_loader):
            print(timestamp(),iteration,"/",len(data_loader)-1,end="\r")

            x = x.to(device)

            recon_x, z,mean,_ = net(x)
            
            x = x.detach().cpu().squeeze(1)
            recon_x = recon_x.detach().cpu().squeeze(1)
            z = z.detach().cpu()
            mean = mean.detach().cpu()

            zs = torch.cat((zs,z),0)
            means = torch.cat((means,mean),0)

            x = torch.round(x*255)
            recon_x = torch.round(recon_x*255)

            xshape = x.shape
            xcode = torch.zeros((*xshape,8))
            for n in range(xshape[0]):
                for r,c in pixels:
                    xcode[n,r,c] = torch.from_numpy(dec2(x[n,r,c]))

            recon_xcode = torch.zeros((*xshape,8))
            for n in range(xshape[0]):
                for r,c in pixels:
                    recon_xcode[n,r,c] = torch.from_numpy(dec2(recon_x[n,r,c]))
                        
            mses = torch.cat((mses,xcode-recon_xcode),0)
        
        subset_recon,_,_,_ = net(subset)

    subset = subset.detach().cpu().squeeze(1).numpy()
    subset_recon = subset_recon.detach().cpu().squeeze(1).numpy()
    zs = zs[1:,:].numpy()
    if args.model == "VAE":
        means = means[1:,:].numpy()
    mses = mses[1:,:,:].numpy()

    for r,c in pixels:
        AnalysisMSE(mses[:,r,c,:],f"../binarymses/{args.model}-{args.latent_dim}-{r}-{c}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis AE on MNIST")
    parser.add_argument("--net_file", type=str, default="results/0224_151045048077_AE.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default="AE")
    parser.add_argument("--latent_dim", type=int, default=8)
    args = parser.parse_args()

    main(args)