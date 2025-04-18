from scapy.all import rdpcap, IP, TCP
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

packets = rdpcap("synthetic_traffic.pcap")

data = []

for pkt in packets:
    if IP in pkt and TCP in pkt:
        data.append({
            "src": pkt[IP].src,
            "dst": pkt[IP].dst,
            "dport": pkt[TCP].dport,
        })

df = pd.DataFrame(data)

df['dport'].plot(kind='hist', bins=50, color='skyblue')
plt.title('Destination Port Distribution')
plt.xlabel('Port')
plt.ylabel('Number of Packets')
plt.grid(True)
plt.tight_layout()
# plt.show()


G = nx.DiGraph()

i = 0
for _, row in df.iterrows():
    G.add_edge(row['src'], row['dst'])
    if i == 10:
        break
    i += 1

plt.figure(figsize=(10, 6))
nx.draw(G, with_labels=True, node_size=700, node_color='lightgreen', arrowsize=10)
plt.title('Synthetic Packet Flow (Source → Destination)')
plt.show()
