{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import numpy as np\n",
    "from typing import List\n",
    "import scipy\n",
    "from scipy import linalg\n",
    "torch.set_default_tensor_type(torch.DoubleTensor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "N = 1000000\n",
    "d = 32\n",
    "l = 5\n",
    "H = (torch.randn(N,d) + torch.rand(N,d)**2)\n",
    "\n",
    "H = H - torch.mean(H, dim = 0, keepdim = True)\n",
    "H = H\n",
    "W = torch.randn(d,l)\n",
    "cov_H = torch.tensor(np.cov(H.detach().cpu().numpy(), rowvar = False))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# u = torch.nn.Parameter(u)\n",
    "# optimizer = torch.optim.SGD([u], lr=0.001, momentum=0.9, weight_decay=1e-4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_cov_output_projected(u, cov_H, W):\n",
    "    \n",
    "    u_normed = u / torch.norm(u)\n",
    "    P = torch.eye(cov_H.shape[0]) - (u_normed@u_normed.T)\n",
    "    #P = u_normed@u_normed.T\n",
    "    return W.T@P@cov_H@P@W\n",
    "\n",
    "def get_cov_output_total(H,W):\n",
    "    Y_hat = H@W \n",
    "    return torch.tensor(np.cov(Y_hat.detach().cpu().numpy(), rowvar = False))\n",
    "\n",
    "def get_loss_func(cov_output_projected):\n",
    "    \n",
    "    return torch.sum(torch.diag(cov_output_projected))\n",
    "\n",
    "def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):\n",
    "    \"\"\"\n",
    "    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),\n",
    "    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.\n",
    "    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:\n",
    "    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))\n",
    "    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections\n",
    "    :param dim: input dim\n",
    "    \"\"\"\n",
    "\n",
    "    I = np.eye(input_dim)\n",
    "    Q = np.sum(rowspace_projection_matrices, axis = 0)\n",
    "    P = I - get_rowspace_projection(Q)\n",
    "\n",
    "    return P\n",
    "\n",
    "def get_rowspace_projection(W: np.ndarray) -> np.ndarray:\n",
    "    \"\"\"\n",
    "    :param W: the matrix over its nullspace to project\n",
    "    :return: the projection matrix over the rowspace\n",
    "    \"\"\"\n",
    "\n",
    "    if np.allclose(W, 0):\n",
    "        w_basis = np.zeros_like(W.T)\n",
    "    else:\n",
    "        w_basis = scipy.linalg.orth(W.T) # orthogonal basis\n",
    "\n",
    "    P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace\n",
    "\n",
    "    return P_W\n",
    "\n",
    "def BCA(H,W,n_components, eps = 1e-7, max_iters = 25000):\n",
    "    \n",
    "    P_nullspace = torch.eye(H.shape[1])\n",
    "    results = []\n",
    "    cov_out_total = get_cov_output_total(H,W)\n",
    "    total_var = get_loss_func(cov_out_total).detach().cpu().numpy().item()\n",
    "    H_proj = H.clone()\n",
    "    rowspace_projs = []\n",
    "    \n",
    "    for i in range(n_components):\n",
    "        \n",
    "        H_proj = H_proj@P_nullspace # remove previous component \n",
    "        #if i > 0: print(\"test: \", H_proj@u.double())\n",
    "        #print(\"H proj\", H_proj[:10,:])\n",
    "        cov_H = torch.from_numpy(np.cov(H_proj.detach().cpu().numpy(), rowvar = False))\n",
    "        #print(\"COV H proj\", cov_H)\n",
    "        print(\"-----------------------------\")\n",
    "        u = torch.randn(H_proj.shape[1], 1)\n",
    "        u = torch.nn.Parameter(u)\n",
    "        optimizer = torch.optim.SGD([u], lr=1e-3, momentum=0.9, weight_decay=1e-6)\n",
    "        \n",
    "        diff = 10\n",
    "        j = 0\n",
    "        loss_vals = [np.inf]\n",
    "        \n",
    "        while j < max_iters and diff > eps:\n",
    "            optimizer.zero_grad()\n",
    "            cov_out = get_cov_output_projected(u,cov_H,W)\n",
    "            loss = get_loss_func(cov_out)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            loss_vals.append(loss.detach().cpu().numpy().item())\n",
    "            diff = np.abs(loss_vals[-1] - loss_vals[-2])\n",
    "            if j % 500 == 0: print(\"j, loss, \", j, loss.detach().cpu().numpy().item(), diff)\n",
    "            j += 1\n",
    "        print(\"finished after {} iters\".format(j))\n",
    "        \n",
    "        # calculate new nullspace projection to neutralzie component u\n",
    "        \n",
    "        u_normed = u / torch.norm(u)\n",
    "        rowspace_projs.append((u_normed@u_normed.T).detach().cpu().numpy())\n",
    "        #P_nullspace = torch.from_numpy(get_projection_to_intersection_of_nullspaces(rowspace_projs,cov_H.shape[0]))\n",
    "        P_nullspace = torch.eye(H_proj.shape[1]).double() - u_normed@u_normed.T\n",
    "        \n",
    "        # calcualte explained variance\n",
    "        cov_out_projected = get_cov_output_projected(u,cov_H,W)\n",
    "        total_var_projected = total_var-get_loss_func(cov_out_projected).detach().cpu().numpy().item()\n",
    "        explained_var = total_var_projected / total_var\n",
    "        \n",
    "        u = u / u.norm()\n",
    "        results.append({\"vec\": u.squeeze().detach().cpu().numpy(), \"projected_var\": total_var_projected,\n",
    "                       \"total_var\": total_var, \"explained_var\": total_var_projected*100/total_var,\n",
    "                       \"cov_out\":cov_out_projected})\n",
    "    \n",
    "    return results\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-----------------------------\n",
      "j, loss,  0 135.26160273520554 inf\n",
      "j, loss,  500 100.55800708361618 0.0035053699756133483\n",
      "j, loss,  1000 99.9186858463355 0.0005054382567237781\n",
      "j, loss,  1500 99.7617881388654 0.00017777623224901618\n",
      "j, loss,  2000 99.7078965576554 5.859931863483325e-05\n",
      "j, loss,  2500 99.69045049489841 1.862443282618642e-05\n",
      "j, loss,  3000 99.6849391273175 5.847988333584908e-06\n",
      "j, loss,  3500 99.68321197067934 1.8290295429324033e-06\n",
      "j, loss,  4000 99.68267212884751 5.713088597758542e-07\n",
      "j, loss,  4500 99.68250354281327 1.7837221832905925e-07\n",
      "finished after 4750 iters\n",
      "-----------------------------\n",
      "j, loss,  0 99.1276355328844 inf\n",
      "j, loss,  500 63.05948957976347 0.0027859741574047803\n",
      "j, loss,  1000 62.67309401257155 7.575458378994426e-05\n",
      "j, loss,  1500 62.663198902056045 1.8275550885960001e-06\n",
      "finished after 1891 iters\n",
      "-----------------------------\n",
      "j, loss,  0 61.444175393820906 inf\n",
      "j, loss,  500 39.99108873018874 0.030079992229133268\n",
      "j, loss,  1000 32.36511302457802 0.00039493181522232135\n",
      "j, loss,  1500 32.336569538241505 4.07354022513573e-07\n",
      "finished after 1604 iters\n",
      "-----------------------------\n",
      "j, loss,  0 38.05318820225911 inf\n",
      "j, loss,  500 14.097091993039808 0.008651382117115247\n",
      "j, loss,  1000 12.972298202795539 0.000173185460056402\n",
      "j, loss,  1500 12.951841361986222 2.7126522397225017e-06\n",
      "finished after 1898 iters\n",
      "-----------------------------\n",
      "j, loss,  0 12.809223517977209 inf\n",
      "j, loss,  500 0.03347627646670668 0.0006457214496600402\n",
      "finished after 960 iters\n"
     ]
    }
   ],
   "source": [
    "bca = BCA(H,W,n_components=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "#bca[0][\"vec\"].T@bca[3][\"vec\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 1 -1.265984961192762e-06\n",
      "0 2 -1.407897433206018e-06\n",
      "0 3 -1.1746536428269838e-06\n",
      "0 4 -1.6790201201599686e-08\n",
      "1 2 -5.058391205903234e-10\n",
      "1 3 -4.4132394230039784e-07\n",
      "1 4 -8.326525940960394e-09\n",
      "2 3 -1.1876170933067254e-06\n",
      "2 4 -2.227892656470054e-08\n",
      "3 4 9.776601021804776e-09\n"
     ]
    }
   ],
   "source": [
    "vecs = np.array([x[\"vec\"] for x in bca])\n",
    "for i,v in enumerate(vecs):\n",
    "    for j, v2 in enumerate(vecs):\n",
    "        if j <= i: continue\n",
    "            \n",
    "        print(i,j, v@v2.T)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "350.39296051290836\n"
     ]
    }
   ],
   "source": [
    "\n",
    "#for i in range(len(bca)):\n",
    "    \n",
    "    #print(bca[i][\"explained_var\"])\n",
    "    \n",
    "    \n",
    "print(sum([x[\"explained_var\"] for x in bca]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 4.0669, -3.6262, -0.9620,  1.9986,  4.2501],\n",
       "        [-3.6262,  3.2333,  0.8578, -1.7820, -3.7896],\n",
       "        [-0.9620,  0.8578,  0.2276, -0.4728, -1.0054],\n",
       "        [ 1.9986, -1.7820, -0.4728,  0.9822,  2.0886],\n",
       "        [ 4.2501, -3.7896, -1.0054,  2.0886,  4.4416]], grad_fn=<MmBackward>)"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bca[-2][\"cov_out\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "81.3286209252445"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bca[1][\"explained_var\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8.559765125870644e-09"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bca[0][\"vec\"].T@bca[2][\"vec\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "ename": "RuntimeError",
     "evalue": "size mismatch, m1: [1000000 x 32], m2: [2 x 2] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:41",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-12-824f8956ce8f>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0mP_u1\u001b[0m\u001b[0;34m=\u001b[0m \u001b[0mP_u1\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      8\u001b[0m \u001b[0mP_u2\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mP_u2\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 9\u001b[0;31m \u001b[0mH\u001b[0m\u001b[0;34m@\u001b[0m\u001b[0mP_u1\u001b[0m\u001b[0;34m@\u001b[0m\u001b[0mP_u2\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mRuntimeError\u001b[0m: size mismatch, m1: [1000000 x 32], m2: [2 x 2] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:41"
     ]
    }
   ],
   "source": [
    "H.shape\n",
    "u1, u2 = torch.randn(2,1), torch.randn(2,1)\n",
    "u1 = u1 / torch.norm(u1)\n",
    "u2 = u2 / torch.norm(u2)\n",
    "P_u1 = P_u1 = torch.eye(2) - u1@u1.T\n",
    "P_u2 = torch.eye(2) - u2@u2.T\n",
    "P_u1= P_u1\n",
    "P_u2 = P_u2\n",
    "H@P_u1@P_u2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "P_u1 = torch.eye(2) - u1@u1.T\n",
    "P_u2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "torch.allclose"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
