# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

from json import load
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

ps = load(open('result_stereo1.edex'))[1]["points3d"]
ax = plt.figure().add_subplot(1, 1, 1, projection='3d')
ax.scatter([ps[i][0] for i in ps],
           [ps[i][2] for i in ps],
           [ps[i][1] for i in ps], marker="o", s=2)
scale = 30
ax.set_zlim(-scale, scale)
plt.xlim(-scale, scale)
plt.ylim(-scale, scale)

plt.show()
