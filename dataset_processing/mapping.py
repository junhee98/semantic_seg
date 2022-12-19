import numpy as np
from PIL import Image
import cv2
import os
from tqdm import tqdm

# bdd: city
bdd = {255:0,
0:7,
1:8,
2:11,
4:13,
3:12,
5:17,
6:19,
7:20,
9:22,
8:21,
10:23,
11:24,
12:25,
18:33,
15:28,
13:26,
17:32,
16:31,
14:27,
}

# map: city
map = {13:7,
24:7,
41:7,
2:8,
15:8,
17:11,
6:12,
3:13,
45:17,
47:17,
48:19,
50:20,
30:21,
29:22,
27:23,
19:24,
20:25,
21:25,
22:25,
55:26,
61:27,
54:28,
58:31,
57:32,
52:33,
0:0,
1:0,
4:0,
5:0,
7:0,
8:0,
9:0,
10:0,
11:0,
12:0,
14:0,
16:0,
18:0,
23:0,
25:0,
26:0,
28:0,
31:0,
32:0,
33:0,
34:0,
35:0,
36:0,
37:0,
38:0,
39:0,
40:0,
42:0,
43:0,
44:0,
46:0,
49:0,
51:0,
53:0,
56:0,
59:0,
60:0,
62:0,
63:0,
64:0,
65:0,
}

# cam: city
cam={0:0,
1:15,
2:25,
3:15,
4:11,
5:26,
6:0,
7:24,
8:17,
9:13,
10:7,
11:7,
12:20,
13:32,
14:0,
15:0,
16:24,
17:7,
18:7,
19:8,
20:20,
21:23,
22:27,
23:0,
24:19,
25:31,
26:21,
27:28,
28:16,
29:21,
30:0,
31:12,
32:0,
}


def replace_with_dict2(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    return vs[np.searchsorted(ks,ar)]

def replace_with_dict2_generic(ar, dic, assume_all_present=True):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    idx = np.searchsorted(ks,ar)

    if assume_all_present==0:
        idx[idx==len(vs)] = 0
        mask = ks[idx] == ar
        return np.where(mask, vs[idx], ar)
    else:
        return vs[idx]



if __name__=='__main__':
    path_list = ['/home/awesome/modify_code/pre/bdd/train',
                '/home/awesome/modify_code/pre/bdd/val',
                '/home/awesome/modify_code/pre/map/train',
                '/home/awesome/modify_code/pre/map/val',
                '/home/awesome/modify_code/pre/CamVid/train',
                '/home/awesome/modify_code/pre/CamVid/val',
                '/home/awesome/modify_code/pre/CamVid/test',]
    save_path_list = ['/home/awesome/modify_code/post/bdd/train',
                    '/home/awesome/modify_code/post/bdd/val',
                    '/home/awesome/modify_code/post/map/train',
                    '/home/awesome/modify_code/post/map/val',
                    '/home/awesome/modify_code/post/CamVid/train',
                    '/home/awesome/modify_code/post/CamVid/val',
                    '/home/awesome/modify_code/post/CamVid/test',]
    dict_list=[bdd,bdd,map,map,cam,cam,cam]

    for path, save_path, dict in zip(path_list, save_path_list,dict_list):
        img_list = os.listdir(path)

        for img in tqdm(img_list):
            n1 = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            out = replace_with_dict2(n1, dict)
            cv2.imwrite(os.path.join(save_path,img),out)
