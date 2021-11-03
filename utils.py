import numpy,torch,datetime
import matlab.engine
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from itertools import product

def __dec2__(i,s,dtype = "binary"):
    if dtype == "binary":
        divisor = 2
    remainder,quotient = i%divisor,i//divisor
    if quotient == 0:
        return [remainder]
    else:
        return __dec2__(quotient,s,dtype) + [remainder] + s

def dec2(i,padding = 8, dtype = "binary"):
    code = __dec2__(i,[],dtype)
    code = (padding-len(code))*[0]+code
    return numpy.array(code)

# numpy.array(list(bin(200)[2:]),dtype = float)

def __to__(i, dtype = "int", to = "binary"):
    if dtype == "int" and to == "binary":
        return numpy.array(list(bin(i)[2:]),dtype = float)
    if dtype == "binary" and to == "int":
        return int("".join(i.astype(int).astype(str)),2)

def to(i,padding = 8, dtype = "int", to = "binary"):
    code = __to__(i, dtype= dtype, to = to)
    if isinstance(code,numpy.ndarray):
        code = numpy.pad(code,(8-len(code),0),constant_values = 0)
    return code

# int("".join(b.astype(int).astype(str)),2)

def writerscalar(writer, scalar_dict, step):
    for k,v in scalar_dict.items():
        if isinstance(v,(int,float)):
            writer.add_scalar(k,v,step)

def writeimage(writer, tag, tensor, step):
    img = make_grid(tensor)
    writer.add_image(tag, img, step)

class BinDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        start_time = timestamp()
        self.__initdata(data)
        print(f"Binaried, start at {start_time}, end at {timestamp()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def __initdata(self,data):
        self.data = []
        for ind,(img,label) in enumerate(data):
            print("Binarizing: ",ind, end="\r")
            tensor = numpy.zeros((*img.shape,8))
            for x in range(28):
                for y in range(28):
                    tensor[0,x,y,:] =  to(img[0,x,y].int().item(), dtype="int", to= "binary")
            self.data.append([img,tensor,label])

def tensor_int2bin(t):
    t = (t * 255).to(torch.int32)
    t_binary = torch.zeros(*t.shape,8,device=t.device)
    for i in range(1,9):
        t, remainder  = t//2, t % 2
        t_binary[:,:,:,:,-i] = remainder
    return t_binary.to(torch.float32)

def tensor_bin2int(t_binary):
    t_binary = t_binary.to(torch.int32)
    t = torch.zeros(*(t_binary.shape[:-1]),device=t_binary.device)
    for i in range(1,9):
        t = t + 2**(i-1)*t_binary[:,:,:,:,-i]
    t = t/255
    return t.to(torch.float32)

def tensor_hamming_encode(t_binary):
    hamming_code = torch.zeros(*(t_binary.shape[:-1]),6, device= t_binary.device)
    hamming_code[:,:,:,:,0] = (t_binary[:,:,:,:,0]+t_binary[:,:,:,:,1]+t_binary[:,:,:,:,3])%2
    hamming_code[:,:,:,:,1] = (t_binary[:,:,:,:,0]+t_binary[:,:,:,:,2]+t_binary[:,:,:,:,3])%2
    hamming_code[:,:,:,:,2] = (t_binary[:,:,:,:,1]+t_binary[:,:,:,:,2]+t_binary[:,:,:,:,3])%2
    hamming_code[:,:,:,:,3] = (t_binary[:,:,:,:,4]+t_binary[:,:,:,:,5]+t_binary[:,:,:,:,7])%2
    hamming_code[:,:,:,:,4] = (t_binary[:,:,:,:,4]+t_binary[:,:,:,:,6]+t_binary[:,:,:,:,7])%2
    hamming_code[:,:,:,:,5] = (t_binary[:,:,:,:,5]+t_binary[:,:,:,:,6]+t_binary[:,:,:,:,7])%2
    return hamming_code

def tensor_hamming_decode(t_binary,hamming_code):
    shape = t_binary.shape
    index_0 = torch.arange(0,shape[0]).reshape(1,-1,1).expand(1,shape[0],shape[1]*shape[2]*shape[3]).flatten().long()
    index_1 = torch.arange(0,shape[1]).reshape(1,-1,1).expand(shape[0],shape[1],shape[2]*shape[3]).flatten().long()
    index_2 = torch.arange(0,shape[2]).reshape(1,-1,1).expand(shape[0]*shape[1],shape[2],shape[3]).flatten().long()
    index_3 = torch.arange(0,shape[3]).reshape(1,-1,1).expand(shape[0]*shape[1]*shape[2],shape[3],1).flatten().long()
    loc = torch.zeros(*(shape[:-1]), device=t_binary.device)
    loc = loc + (hamming_code[:,:,:,:,0]+t_binary[:,:,:,:,0]+t_binary[:,:,:,:,1]+t_binary[:,:,:,:,3])%2
    loc = loc + 2*((hamming_code[:,:,:,:,1]+t_binary[:,:,:,:,0]+t_binary[:,:,:,:,2]+t_binary[:,:,:,:,3])%2)
    loc = loc + 4*((hamming_code[:,:,:,:,2]+t_binary[:,:,:,:,1]+t_binary[:,:,:,:,2]+t_binary[:,:,:,:,3])%2)
    loc_ori = loc.clone()
    loc[loc_ori==3] = 1
    loc[loc_ori==5] = 2
    loc[loc_ori==6] = 3
    loc[loc_ori==7] = 4
    mask = torch.zeros(*(shape[:-1]),shape[-1]+1, device=t_binary.device)
    mask[index_0,index_1,index_2,index_3,loc.flatten().long()] = 1
    t_binary = (t_binary + mask[:,:,:,:,1:])%2
    loc = torch.zeros(*(shape[:-1]), device=t_binary.device)
    loc = loc + (hamming_code[:,:,:,:,3]+t_binary[:,:,:,:,4]+t_binary[:,:,:,:,5]+t_binary[:,:,:,:,7])%2
    loc = loc + 2*((hamming_code[:,:,:,:,4]+t_binary[:,:,:,:,4]+t_binary[:,:,:,:,6]+t_binary[:,:,:,:,7])%2)
    loc = loc + 4*((hamming_code[:,:,:,:,5]+t_binary[:,:,:,:,5]+t_binary[:,:,:,:,6]+t_binary[:,:,:,:,7])%2)
    loc_ori = loc.clone()
    loc[loc_ori==3] = 5
    loc[loc_ori==5] = 6
    loc[loc_ori==6] = 7
    loc[loc_ori==7] = 8
    mask = torch.zeros(*(shape[:-1]),shape[-1]+1, device=t_binary.device)
    mask[index_0,index_1,index_2,index_3,loc.flatten().long()] = 1
    t_binary = (t_binary + mask[:,:,:,:,1:])%2
    return t_binary

def tensor_squarehamming_encode(t_binary_first):
    hamming_code = torch.zeros_like(t_binary_first, device= t_binary_first.device)
    for r in range(0,28,2):
        for c in range(0,28,2):
            hamming_code[:,:,r,c] = (t_binary_first[:,:,r,c]+t_binary_first[:,:,r,c+1]+t_binary_first[:,:,r+1,c+1])%2
            hamming_code[:,:,r,c+1] = (t_binary_first[:,:,r,c]+t_binary_first[:,:,r+1,c]+t_binary_first[:,:,r+1,c+1])%2
            hamming_code[:,:,r+1,c] = (t_binary_first[:,:,r,c+1]+t_binary_first[:,:,r+1,c]+t_binary_first[:,:,r+1,c+1])%2
    return hamming_code

def tensor_squarehamming_decode(recon_t_binary_first,hamming_code):
    mask = torch.zeros_like(recon_t_binary_first, device= recon_t_binary_first.device).float()
    for r in range(0,28,2):
        for c in range(0,28,2):
            loc = torch.zeros(*(recon_t_binary_first.shape[:2]), device= recon_t_binary_first.device)
            loc = loc + (hamming_code[:,:,r,c]+recon_t_binary_first[:,:,r,c]+recon_t_binary_first[:,:,r,c+1]+recon_t_binary_first[:,:,r+1,c+1])%2
            loc = loc + 2*((hamming_code[:,:,r,c+1]+recon_t_binary_first[:,:,r,c]+recon_t_binary_first[:,:,r+1,c]+recon_t_binary_first[:,:,r+1,c+1])%2)
            loc = loc + 4*((hamming_code[:,:,r+1,c]+recon_t_binary_first[:,:,r,c+1]+recon_t_binary_first[:,:,r+1,c]+recon_t_binary_first[:,:,r+1,c+1])%2)
            mask[:,:,r:r+2,c:c+2][loc==3] = torch.tensor([[1,0],[0,0]], device= recon_t_binary_first.device).float()
            mask[:,:,r:r+2,c:c+2][loc==5] = torch.tensor([[0,1],[0,0]], device= recon_t_binary_first.device).float()
            mask[:,:,r:r+2,c:c+2][loc==6] = torch.tensor([[0,0],[1,0]], device= recon_t_binary_first.device).float()
            mask[:,:,r:r+2,c:c+2][loc==7] = torch.tensor([[0,0],[0,1]], device= recon_t_binary_first.device).float()
    recon_t_binary_first = recon_t_binary_first + mask
    return recon_t_binary_first % 2

def hamming_recon(t_binary,recon_t_binary):
    t,recon_t = tensor_bin2int(t_binary) , tensor_bin2int(recon_t_binary) 
    hamming_code = tensor_hamming_encode(t_binary)
    hamming_recon_t_binary = tensor_hamming_decode(recon_t_binary,hamming_code)
    hamming_recon_t = tensor_bin2int(hamming_recon_t_binary)
    return hamming_recon_t, hamming_recon_t_binary, t,recon_t

def squarehamming_recon(t_binary,recon_t_binary, indexs = [0]):
    for index in indexs:
        t_binary_first, recon_t_binary_first =t_binary[:,:,:,:,index],recon_t_binary[:,:,:,:,index]
        hamming_code = tensor_squarehamming_encode(t_binary_first)
        hamming_recon_t_binary_first = tensor_squarehamming_decode(recon_t_binary_first,hamming_code)
        hamming_recon_t_binary = recon_t_binary.clone()
        hamming_recon_t_binary[:,:,:,:,index] = hamming_recon_t_binary_first
        hamming_recon_t_real = tensor_bin2int(hamming_recon_t_binary)
    return hamming_recon_t_binary, hamming_recon_t_real

def __ReedSolomonRecon__(eng, t,recon_t,m,n,k):
    t_flatten,recon_t_flatten = torch.zeros(len(t),k),torch.zeros(len(recon_t),k)
    # t, recon_t = t.reshape(len(t),-1), recon_t.reshape(len(recon_t),-1)
    t_flatten[:,:t.shape[1]],recon_t_flatten[:,:recon_t.shape[1]] = t, recon_t
    tf_matlab,recon_tf_matlab = matlab.double(t_flatten.tolist()),matlab.double(recon_t_flatten.tolist())
	# m_matlab, n_matlab, k_matlab = matlab.int32(m), matlab.int32(n), matlab.int32(k)
	# recon_t_rs, cnumerr = eng.ReedSolomonRecon(tf_matlab, recon_tf_matlab, m_matlab, n_matlab, k_matlab)
    recon_t_rs, cnumerr = eng.ReedSolomonRecon(tf_matlab, recon_tf_matlab, m, n, k, nargout = 2)
    recon_t_rs, cnumerr = torch.tensor(recon_t_rs), torch.tensor(cnumerr)
    recon_t_rs = recon_t_rs[:,:t.shape[1]]
    return recon_t_rs

def ReedSolomonRecon(eng, t,recon_t,err_ig = 90,err_scale = 3,sym_count = 4,sym_size = 2,code_len = 250,msg_len = 200,img_shape = 512*768):

    com_len = int(img_shape/sym_count)
    padding_len = int(numpy.ceil(com_len/msg_len)*msg_len)
    t_ori = t.clone()
    recon_t_ori = recon_t.clone()
    t = t.clone()
    recon_t = recon_t.clone()
    cutpoint = list(range(0,255,err_ig))
    # cutpoint = [0,err_ig]
    for e,(st,ed) in enumerate(zip(cutpoint,cutpoint[1:]+[256])):
        t[(t_ori>=st) & (t_ori<ed)] = e
        recon_t[(recon_t_ori>=st) & (recon_t_ori<ed)] = e
    t = t%err_scale
    recon_t = recon_t%err_scale
    t_flatten = t.flatten()
    recon_t_flatten = recon_t.flatten()
    t_com_padding = torch.zeros(padding_len)
    recon_t_com_padding = torch.zeros(padding_len)

    for i in range(sym_count):
        t_com_padding[:com_len] += ((2**sym_size)**i)*t_flatten[i::sym_count]
        recon_t_com_padding[:com_len] += ((2**sym_size)**i)*recon_t_flatten[i::sym_count]

    t_remat = t_com_padding.reshape(-1,msg_len)
    recon_t_remat = recon_t_com_padding.reshape(-1,msg_len)
    recon_t_remat_rs = __ReedSolomonRecon__(eng, t_remat, recon_t_remat, sym_size*sym_count, code_len, msg_len)
    recon_t_com_rs = recon_t_remat_rs.flatten()[:com_len]

    recon_t_flatten_rs = torch.zeros_like(recon_t_flatten)
    for i in range(sym_count):
        recon_t_flatten_rs[i::sym_count] = recon_t_com_rs % (2**sym_size)
        recon_t_com_rs = recon_t_com_rs // (2**sym_size)
        
    recon_t_final = recon_t_flatten_rs.reshape(*recon_t_ori.shape)

    return recon_t_final