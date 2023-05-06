#############   Define the discrete action space   ###############
import numpy as np 

disc_actions = [np.array(-2, dtype=np.float32), 
                np.array(-1.5, dtype=np.float32),
                np.array(-1, dtype=np.float32),
                np.array(-0.5, dtype=np.float32),
                np.array(0, dtype=np.float32), 
                np.array(0.5, dtype=np.float32),
                np.array(1, dtype=np.float32),
                np.array(1.5, dtype=np.float32),
                np.array(2, dtype=np.float32)]