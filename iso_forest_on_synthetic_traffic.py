from scapy.all import rdpcap, IP, TCP
import pandas as pd

packets = rdpcap("synthetic_traffic.pcap")

data = []

for pkt in packets:
    if IP in pkt and TCP in pkt:
        data.append({
            "src_ip": pkt[IP].src,
            "dst_ip": pkt[IP].dst,
            "dport": pkt[TCP].dport,
            "sport": pkt[TCP].sport,
            "packet_len": len(pkt),
        })

df = pd.DataFrame(data)


from sklearn.preprocessing import LabelEncoder

# Encode IPs as numbers
le = LabelEncoder()
df['src_ip_enc'] = le.fit_transform(df['src_ip'])
df['dst_ip_enc'] = le.fit_transform(df['dst_ip'])

# Drop raw IPs (or keep if needed for reference)
X = df[['src_ip_enc', 'dst_ip_enc', 'sport', 'dport', 'packet_len']]


from sklearn.ensemble import IsolationForest

model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)
df['anomaly'] = model.predict(X)

# Map results: -1 = anomaly, 1 = normal
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

# print(df)  # Show only anomalies


import matplotlib.pyplot as plt

plt.scatter(df.index, df['packet_len'], c=df['anomaly'], cmap='coolwarm', alpha=0.7)
plt.title('Anomaly Detection on Packet Length')
plt.xlabel('Packet Index')
plt.ylabel('Packet Size')
plt.show()
