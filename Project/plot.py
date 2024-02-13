import matplotlib.pyplot as plt
import pandas as pd 

# pd_model1 = pd.read_csv("log\\log_dqn_model1.csv", skipinitialspace=True)
# pd_model2 = pd.read_csv("log\\log_dqn_model2.csv", skipinitialspace=True)
# pd_model3 = pd.read_csv("log\\log_dqn_model3.csv", skipinitialspace=True)
# pd_model4 = pd.read_csv("log\\log_dqn_model4.csv", skipinitialspace=True)
# pd_model5 = pd.read_csv("log\\log_dqn_model5.csv", skipinitialspace=True)
# pd_model6 = pd.read_csv("log\\log_dqn_model6.csv", skipinitialspace=True)

# 1 for baseline
# 2 for explore steps 1000k
# 3 for gamma 0.8
# 4 for gamma 0.9
# 5 for memorysize 100ks
# 6 for memorysize 50k

# plt.plot(pd_model1['Episode'], pd_model1['Reward'], label = r"$\gamma$ = 0.99")
# plt.plot(pd_model4['Episode'], pd_model4['Reward'], label = r"$\gamma$ = 0.90")
# plt.plot(pd_model3['Episode'], pd_model3['Reward'], label = r"$\gamma$ = 0.80")
# plt.legend()
# plt.xlabel("Episode")
# plt.ylabel("Reward")

# plt.plot(pd_model1['Episode'], pd_model1['Loss'], label = r"$\gamma$ = 0.99")
# plt.plot(pd_model4['Episode'], pd_model4['Loss'], label = r"$\gamma$ = 0.90")
# plt.plot(pd_model3['Episode'], pd_model3['Loss'], label = r"$\gamma$ = 0.80")
# plt.legend()
# plt.xlabel("Episode")
# plt.ylabel("Loss")

# plt.plot(pd_model1['Episode'], pd_model1['Loss'], label = r"$Explore Steps$ = 800k")
# plt.plot(pd_model4['Episode'], pd_model2['Loss'], label = r"$Explore Steps$ = 1000k")
# plt.legend()
# plt.xlabel("Episode")
# plt.ylabel("Loss")

# plt.plot(pd_model1['Episode'], pd_model1['Loss'], label = r"Memory Size=200k")
# plt.plot(pd_model5['Episode'], pd_model5['Loss'], label = r"Memory Size=100k")
# plt.plot(pd_model6['Episode'], pd_model6['Loss'], label = r"Memory Size=50k")

# plt.legend()
# plt.xlabel("Episode")
# plt.ylabel("Loss")

# fig = plt.figure()
# ax1 = fig.subplots()
# ax2 = ax1.twinx()

# ax1.plot(pd_model1['Episode'], pd_model1['Reward'], 'g--', label = r"Reward")
# ax2.plot(pd_model1['Episode'], pd_model1['Length'], 'b:', label = r"Length")
# ax1.plot(pd_model1['Episode'], pd_model1['Loss'], 'r', label = r"Loss", )

# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
# ax1.set_xlabel("Episode")
# ax1.set_ylabel("Reward / Loss")
# ax2.set_ylabel("Length")

pd_model1 = pd.read_csv("log\\log_ddpg_model1.csv", skipinitialspace=True)
pd_model2 = pd.read_csv("log\\log_ddpg_model2.csv", skipinitialspace=True)
pd_model3 = pd.read_csv("log\\log_ddpg_model3.csv", skipinitialspace=True)
pd_model4 = pd.read_csv("log\\log_ddpg_model4.csv", skipinitialspace=True)
pd_model5 = pd.read_csv("log\\log_ddpg_model5.csv", skipinitialspace=True)

# 1 for baseline
# 2 for gamma 0.9
# 3 for gamma 0.8
# 4 for memorysize 50k
# 5 for memorysize 100ks

plt.plot(pd_model1['Episode'], pd_model1['Loss'], label = "Memory Size = 200k")
plt.plot(pd_model5['Episode'], pd_model5['Loss'], label = "Memory Size = 100k")
plt.plot(pd_model4['Episode'], pd_model4['Loss'], label = "Memory Size = 50k")
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Loss")

plt.show()