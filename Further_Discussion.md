## 1. Model (Intuition and Formal Definition)

### 1.1 Intuisi model (dengan GNN sejak awal)

Kita mempelajari hubungan antara variabel respon dan kovariat yang **bervariasi secara spasial**.
Namun, kita **tidak ingin**:

* memaksakan bentuk kernel spasial tertentu (Gaussian, bisquare, dsb),
* karena struktur spasial **bisa kompleks** (barrier, anisotropi, klaster).

Sebagai gantinya:

> **Kita membiarkan bobot spasial dipelajari dari data**,
> tetapi **tetap menjaga lokalitas spasial** agar inferensi sah.

Dengan kata lain:

* **GWR tetap model inferensi utama**,
* **GNN hanya berperan sebagai *estimator kernel adaptif***.


### 1.2 Data dan notasi

Kita mengamati (\(n\) observasi independen:

\[
{(Y_i, X_i, U_i)}_{i=1}^n
\]

dengan:

* \(Y_i \in \mathbb{R}\)
  variabel respon (skalar)

* \(X_i \in \mathbb{R}^p\)
  vektor kovariat berdimensi (p)

* \(U_i \in \mathbb{R}^d\)
  lokasi spasial berdimensi (d)

### 1.3 Model struktural (*data-generating process*)

Model sebenarnya diasumsikan:

$$\boxed{Y_i= X_i^\top \beta(U_i) + \varepsilon_i}$$

Penjelasan:

* (\beta(\cdot): \mathbb{R}^d \to \mathbb{R}^p)
  fungsi koefisien spasial yang **halus**

* (\beta(U_i))
  koefisien lokal sebenarnya di lokasi (U_i)

* (\varepsilon_i)
  error acak dengan mean nol

Target inferensi **tetap**:

[
\boxed{
\beta(u_0)
}
]

untuk satu lokasi target tetap (u_0 \in \mathbb{R}^d).

---

### 1.4 Prinsip lokalitas (belum pakai kernel)

Agar (\beta(u_0)) dapat diestimasi, kita perlu:

* hanya menggunakan observasi **cukup dekat** dengan (u_0),
* tetapi **jumlahnya bertambah** saat (n) membesar.

Untuk itu, diperkenalkan **bandwidth** (h = h_n > 0) dan neighborhood:

[
\mathcal N_h(u_0)
=================

{ i : |U_i - u_0| \le h }
]

Dengan syarat asimtotik (akan dipakai nanti):

[
h \to 0,
\qquad
n h^d \to \infty
]

📌 **Sampai titik ini: belum ada kernel, belum ada GNN.**
Ini murni struktur nonparametrik lokal.

---

# PART 2 (REWRITE TOTAL)

## 2. Asumsi Klasik yang Digunakan

Asumsi berikut **bukan CLM global**,
melainkan **versi lokal untuk GWR dengan kernel adaptif**.

---

### (A1) Independensi

[
{(X_i, U_i, \varepsilon_i)}_{i=1}^n
\quad \text{i.i.d.}
]

---

### (A2) Eksogenitas lokal

[
\boxed{
\mathbb E[\varepsilon_i \mid X_i, U_i] = 0
}
]

Ini menjamin bahwa **ketidakpastian hanya berasal dari error**,
bukan dari desain pembobotan.

---

### (A3) Variansi terbatas

[
\mathbb E[\varepsilon_i^2 \mid X_i, U_i] = \sigma^2 < \infty
]

---

### (A4) Momen kovariat terbatas

[
\mathbb E[|X_i|^2] < \infty
]

---

### (A5) Kehalusan koefisien spasial

Fungsi (\beta(u)) memiliki turunan kedua kontinu
di sekitar (u_0).

Ini memungkinkan **ekspansi Taylor lokal**.

---

📌 **Perhatikan**:
Belum ada satu pun asumsi tentang **kernel atau GNN**.
Itu **sengaja**.

---

# PART 3 (REWRITE TOTAL)

## 3. Estimator: Kernel sebagai Fungsi yang Diestimasi oleh GNN

Di sinilah perbedaan fundamental dengan GWR klasik dimulai.

---

## 3.1 Masalah yang harus diselesaikan

Dalam GWR klasik, bobot berbentuk:

[
w_i(u_0)
========

\frac{
K!\left( \dfrac{U_i - u_0}{h} \right)
}{
\sum_j K!\left( \dfrac{U_j - u_0}{h} \right)
}
]

Namun di sini:

* kita **tidak ingin menetapkan (K(\cdot)) secara eksplisit**,
* kita ingin **mempelajari bentuk kernel dari data**.

Maka kita perlukan:

> suatu **fungsi pembobotan adaptif**
> yang **menggantikan kernel**,
> tetapi **tidak merusak struktur GWR**.

---

## 3.2 Objek yang ingin kita estimasi (kernel adaptif)

Kita ingin bobot berbentuk:

[
w_i(u_0)
========

\frac{
\exp\big( s(U_i, X_i, u_0) \big)
\cdot \mathbf 1{|U_i-u_0|\le h}
}{
\sum_{j:|U_j-u_0|\le h}
\exp\big( s(U_j, X_j, u_0) \big)
}
]

Penjelasan:

* (s(\cdot)) adalah **fungsi skor**, belum ditentukan bentuknya
* indikator memastikan **lokalitas keras**
* softmax memastikan:

  * bobot positif
  * jumlah bobot = 1

📌 **Ini sudah memberi petunjuk arsitektur**,
tapi **belum menyebut GNN**.

---

## 3.3 Mengapa fungsi skor (s(\cdot)) harus dipelajari?

Karena:

* hubungan spasial bisa **anisotropik**,
* jarak Euclidean saja tidak cukup,
* informasi graf (keterhubungan wilayah) penting.

Maka:
[
s(\cdot) \quad \text{harus mampu memproses struktur graf}
]

Di sinilah **GNN menjadi pilihan natural**.

---

## 3.4 Definisi formal: GNN sebagai estimator kernel

Kita definisikan sebuah fungsi parametrik:

[
s_\theta(i, u_0)
================

\text{GNN}_\theta
\Big(
\mathcal G_h(u_0),
; \mathbf f_i(u_0)
\Big)
]

dengan:

* (\mathcal G_h(u_0)): graf lokal dengan node (i \in \mathcal N_h(u_0))
* (\mathbf f_i(u_0)): fitur node, **minimal** berisi
  [
  \mathbf f_i(u_0) =
  \big( X_i,; U_i - u_0 \big)
  ]

📌 **Belum ada embedding atau layer disebutkan.**
Kita hanya butuh:

* GNN = fungsi kontinu, terbatas, dan invariant terhadap permutasi node.

---

## 3.5 Bobot akhir (kernel hasil estimasi GNN)

Bobot lokal didefinisikan sebagai:

[
\boxed{
w_i(u_0)
========

\frac{
\exp\big( s_\theta(i,u_0) \big)
}{
\sum_{j\in \mathcal N_h(u_0)}
\exp\big( s_\theta(j,u_0) \big)
}
}
]

Inilah **kernel adaptif yang diestimasi dari data**.

---

## 3.6 Estimator GWR dengan kernel hasil GNN

Definisikan matriks bobot:

[
W_\theta(u_0)
=============

\mathrm{diag}\big(w_1(u_0),\dots,w_n(u_0)\big)
]

Estimator koefisien lokal:

[
\boxed{
\hat\beta(u_0)
==============

\big(X^\top W_\theta(u_0) X\big)^{-1}
X^\top W_\theta(u_0) Y
}
]

📌 **Inilah estimator yang akan kita analisis.**

---

## Penutup (sebelum lanjut ke teori)

Sampai titik ini:

* GNN **sudah digunakan secara formal**,
* kernel **sudah digantikan oleh fungsi terestimasi**,
* **belum ada satu pun asumsi tentang arsitektur GNN**.

➡️ **Itu sengaja.**

Karena:

* di **Part 4**, saat kita mencoba membuktikan **konsistensi**,
* **kita akan mentok**,
* dan **di situlah asumsi & desain GNN akan DIPAKSA muncul**
  (boundedness, smoothness, cross-fitting, dsb).

---

Jika ini sudah sesuai dengan yang kamu minta, jawab saja:

> **“lanjut ke konsistensi”**

dan aku lanjut ke:

**PART 4 — Penurunan Konsistensi Koefisien Lokal (dengan kernel hasil GNN, langkah demi langkah, tanpa loncatan)**

Baik. Kita **lanjut ke PART 4: konsistensi**, **pelan**, **baris demi baris**, dan **jujur secara matematis**.
Di bagian ini, **kita akan benar-benar “mentok”**, lalu **asumsi dan desain GNN akan muncul secara terpaksa** (bukan hiasan).

---

# PART 4

## 4. Penurunan Konsistensi Koefisien Lokal

*(dengan kernel yang diestimasi oleh GNN)*

---

## 4.1 Tujuan formal

Yang ingin kita buktikan adalah:

[
\boxed{
\hat\beta(u_0)
;\xrightarrow{p};
\beta(u_0)
}
]

artinya:

> estimator koefisien lokal berbasis GWR
> dengan **kernel hasil GNN**
> adalah **konsisten** untuk koefisien lokal sebenarnya.

---

## 4.2 Mulai dari definisi estimator (tidak ada trik)

Estimator didefinisikan sebagai:

[
\hat\beta(u_0)
==============

\big(X^\top W_\theta(u_0) X\big)^{-1}
X^\top W_\theta(u_0) Y
]

dengan:

* (W_\theta(u_0)) matriks diagonal berisi bobot
* bobot:
  [
  w_i(u_0)
  =
  \frac{
  \exp\big(s_\theta(i,u_0)\big)
  }{
  \sum_{j\in\mathcal N_h(u_0)}
  \exp\big(s_\theta(j,u_0)\big)
  }
  ]

---

## 4.3 Substitusi model struktural (langkah wajib)

Ingat model sebenarnya:

[
Y = X\beta(U) + \varepsilon
]

Substitusi ke estimator:

[
\hat\beta(u_0)
==============

(X^\top W X)^{-1}
X^\top W (X\beta(U) + \varepsilon)
]

Pisahkan:

# [

(X^\top W X)^{-1}
X^\top W X\beta(U)
+
(X^\top W X)^{-1}
X^\top W \varepsilon
]

Sampai sini **belum ada probabilitas**.
Ini **identitas aljabar murni**.

---

## 4.4 Trik fundamental: tambah–kurang (\beta(u_0))

Tuliskan untuk setiap observasi:

[
\beta(U_i)
==========

\beta(u_0)
+
\big(\beta(U_i) - \beta(u_0)\big)
]

Maka:

[
X^\top W X\beta(U)
==================

X^\top W X\beta(u_0)
+
X^\top W X\big(\beta(U)-\beta(u_0)\big)
]

Substitusi kembali:

[
\hat\beta(u_0)
==============

\beta(u_0)
+
\underbrace{
(X^\top W X)^{-1}
X^\top W X\big(\beta(U)-\beta(u_0)\big)
}*{\text{Bias term}}
+
\underbrace{
(X^\top W X)^{-1}
X^\top W \varepsilon
}*{\text{Noise term}}
]

📌 **Inilah dekomposisi utama.**
Semua analisis konsistensi bergantung pada dua suku ini.

---

## 4.5 Strategi pembuktian konsistensi

Untuk menunjukkan:

[
\hat\beta(u_0) \xrightarrow{p} \beta(u_0)
]

cukup menunjukkan bahwa:

1. **Bias term**
   [
   (X^\top W X)^{-1}
   X^\top W X(\beta(U)-\beta(u_0))
   \xrightarrow{p} 0
   ]

2. **Noise term**
   [
   (X^\top W X)^{-1}
   X^\top W \varepsilon
   \xrightarrow{p} 0
   ]

Kita kerjakan satu-satu.

---

# 4.6 Analisis Bias Term (bagian deterministik)

### 4.6.1 Gunakan kehalusan (\beta(\cdot))

Dari Asumsi (A5), (\beta(\cdot)) dua kali terdiferensiasi.

Maka, untuk (U_i) dekat (u_0):

[
\beta(U_i)
==========

\beta(u_0)
+
\nabla\beta(u_0)^\top (U_i-u_0)
+
R_i
]

dengan remainder:
[
|R_i|
\le
C |U_i-u_0|^2
]

---

### 4.6.2 Substitusi ke bias term

Bias term menjadi:

[
(X^\top W X)^{-1}
\sum_{i=1}^n
w_i X_i X_i^\top
\Big[
\nabla\beta(u_0)^\top (U_i-u_0)
+
R_i
\Big]
]

Pisahkan dua bagian:

1. **Suku linear**
2. **Suku remainder kuadrat**

---

### 4.6.3 Kenapa suku linear hilang?

Kita ingin:

[
\sum_i w_i (U_i-u_0) \approx 0
]

Agar ini benar, **kita mentok** dan harus membuat **asumsi/desain**.

---

## 🔴 TITIK KEBUNTUAN PERTAMA

### (di sinilah desain kernel/GNN DIPAKSA muncul)

Jika bobot **tidak simetris** terhadap (u_0),
maka:

[
\sum_i w_i (U_i-u_0) \neq 0
]

➡️ Bias menjadi orde (h),
➡️ Konsistensi tetap bisa, tapi sangat buruk.

### Maka kita BUTUH:

### **Asumsi (K1) — Local symmetry (implisit)**

Secara formal:
[
\sum_{i\in\mathcal N_h(u_0)}
w_i (U_i-u_0)
\xrightarrow{p}
0
]

---

### Bagaimana ini DICAPAI dengan GNN?

➡️ **Desain, bukan asumsi kosong**:

* input GNN memakai **(U_i - u_0)** (koordinat relatif)
* graf dibangun simetris di sekitar (u_0)
* output di-softmax → bobot ter-normalisasi

Ini membuat **ekspektasi bobot simetris** meskipun bentuk kernel fleksibel.

📌 **Catatan penting**
Kita TIDAK mengasumsikan GNN simetris sempurna,
hanya bahwa **dalam limit**, bobot tidak bias arah.

---

### 4.6.4 Remainder term

Karena:

* (|U_i-u_0|\le h),
* (|R_i| \le C h^2),

maka:

[
\Big|
(X^\top W X)^{-1}
\sum_i w_i X_i X_i^\top R_i
\Big|
=====

O_p(h^2)
]

Dan karena (h \to 0):

[
O_p(h^2) \xrightarrow{} 0
]

---

### 4.6.5 Kesimpulan bias term

[
\boxed{
\text{Bias term}
\xrightarrow{p}
0
}
]

---

# 4.7 Analisis Noise Term (bagian stokastik)

Sekarang bagian **paling berbahaya**.

[
(X^\top W X)^{-1}
X^\top W \varepsilon
]

---

## 4.7.1 Masalah utama

Bobot (W) **dihasilkan oleh GNN dari data**.

Maka secara umum:

[
\mathbb E[X^\top W \varepsilon] \neq 0
]

➡️ LLN gagal
➡️ Konsistensi gagal

---

## 🔴 TITIK KEBUNTUAN KEDUA

### (ini yang MEMAKSA cross-fitting)

Tanpa intervensi, estimator **TIDAK konsisten**.

---

## 4.7.2 Solusi: Cross-fitting (desain, bukan asumsi)

### Konstruksi

Bagi data:
[
{1,\dots,n} = I_1 \cup I_2
]

* Gunakan (I_1) untuk **melatih GNN**
* Gunakan (I_2) untuk **menghitung (\hat\beta(u_0))**

Bobot menjadi:
[
W^{(1)}_\theta(u_0)
\quad\text{independen dari}\quad
\varepsilon_i,; i\in I_2
]

---

### Konsekuensi kunci

Kondisional pada (W^{(1)}):

[
\mathbb E\big[X_i w_i \varepsilon_i \mid W^{(1)}\big] = 0
]

➡️ **LLN berlaku kembali**

---

### 4.7.3 Konvergensi noise term

Karena:

* (w_i) bounded dan ter-normalisasi,
* (X_i \varepsilon_i) punya mean nol dan variansi terbatas,

maka:

[
X^\top W \varepsilon
====================

O_p\big(\sqrt{n h^d}\big)
]

dan:

[
(X^\top W X)^{-1}
=================

O_p\big((n h^d)^{-1}\big)
]

Sehingga:

[
(X^\top W X)^{-1}
X^\top W \varepsilon
====================

O_p\big((n h^d)^{-1/2}\big)
\xrightarrow{p} 0
]

karena:
[
n h^d \to \infty
]

---

## 4.8 Kesimpulan Konsistensi

Gabungkan:

* bias term → 0
* noise term → 0

maka:

[
\boxed{
\hat\beta(u_0)
\xrightarrow{p}
\beta(u_0)
}
]

---

## 4.9 Ringkasan logika konsistensi (satu napas)

> Konsistensi koefisien lokal tercapai karena neighborhood menyempit, fungsi koefisien halus, bobot GNN bersifat lokal dan simetris secara limit, serta cross-fitting memulihkan eksogenitas antara bobot dan error.

---

📌 **STOP DI SINI DULU.**

Bagian berikutnya (**PART 5**) akan:

* menurunkan **variansi koefisien lokal**
* tanpa CLT dulu
* murni second-moment calculation

Jika siap, bilang saja:

> **“lanjut ke variansi”**

Baik. Kita **lanjut ke PART 5: variansi koefisien lokal**, **tanpa CLT**, **tanpa shortcut**, dan **setiap langkah dijelaskan**.
Tujuan bagian ini **bukan distribusi**, melainkan **ukuran penyebaran** dari estimator—ini akan jadi bahan bakar PART 6 (CLT).

---

# PART 5

## 5. Penurunan Variansi Koefisien Lokal

*(GWR dengan kernel hasil GNN, cross-fitted)*

---

## 5.1 Tujuan formal bagian ini

Kita ingin menghitung (atau membatasi orde dari):

[
\mathrm{Var}!\big(\hat\beta(u_0)\big)
]

lebih tepatnya, **orde asimtotiknya** sebagai fungsi dari:

* ukuran sampel (n),
* bandwidth (h),
* dimensi spasial (d).

Ini penting karena:

* menentukan **skala normalisasi** di CLT,
* menentukan **lebar confidence interval**,
* dan membedakan **noise** vs **bias**.

---

## 5.2 Ingat kembali dekomposisi (hasil PART 4)

Dari PART 4, kita punya:

[
\hat\beta(u_0)
==============

\beta(u_0)
+
\underbrace{B_n(u_0)}*{\text{bias}}
+
\underbrace{
(X^\top W X)^{-1} X^\top W \varepsilon
}*{\text{noise}}
]

Variansi hanya datang dari **noise term** karena:

* (\beta(u_0)) deterministik,
* (B_n(u_0)) deterministik bersyarat pada (U) (dan kecil).

Jadi:

[
\mathrm{Var}!\big(\hat\beta(u_0)\big)
=====================================

\mathrm{Var}!\Big(
(X^\top W X)^{-1} X^\top W \varepsilon
\Big)
]

---

## 5.3 Notasi yang akan kita pakai (dijelaskan dulu)

Agar tidak membingungkan, kita perjelas:

* (W = W_\theta(u_0)): matriks bobot diagonal
  dengan elemen (w_i = w_i(u_0))

* (X): matriks desain (n \times p)

* (\varepsilon = (\varepsilon_1,\dots,\varepsilon_n)^\top)

* Kita bekerja **kondisional pada bobot (W)**
  (ini sah karena cross-fitting)

---

## 5.4 Variansi bersyarat (langkah aman)

Karena cross-fitting membuat (W) **independen dari (\varepsilon)**
pada fold estimasi, kita boleh menulis:

[
\mathrm{Var}!\big(\hat\beta(u_0)\big)
=====================================

\mathbb E!\left[
\mathrm{Var}!\Big(
(X^\top W X)^{-1} X^\top W \varepsilon
;\Big|; W, X
\Big)
\right]
]

Kita hitung bagian dalam terlebih dahulu.

---

## 5.5 Hitung variansi kondisional (aljabar matriks)

Gunakan fakta umum:

Jika (A) matriks deterministik dan
(\mathrm{Var}(\varepsilon \mid X,U)=\sigma^2 I_n), maka:

[
\mathrm{Var}(A\varepsilon) = \sigma^2 A A^\top
]

Di sini:
[
A = (X^\top W X)^{-1} X^\top W
]

Maka:

[
\mathrm{Var}!\Big(
(X^\top W X)^{-1} X^\top W \varepsilon
;\Big|; W, X
\Big)
=====

\sigma^2
(X^\top W X)^{-1}
X^\top W^2 X
(X^\top W X)^{-1}
]

📌 **Ini ekspresi variansi eksak** (belum asimtotik).

---

## 5.6 Sekarang kita cari orde asimtotiknya

Agar tidak lompat, kita analisis **tiap faktor**.

---

### 5.6.1 Orde dari (X^\top W X)

Perhatikan:

[
X^\top W X
==========

\sum_{i=1}^n
w_i X_i X_i^\top
]

Karena:

* bobot (w_i) hanya non-nol di neighborhood (\mathcal N_h(u_0)),
* ukuran neighborhood (#\mathcal N_h(u_0) \asymp n h^d),

dan bobot **ter-normalisasi** ((\sum w_i=1)),
maka secara orde:

[
X^\top W X
==========

O_p(n h^d)
]

Lebih tepatnya:
[
\frac{1}{n h^d} X^\top W X
\xrightarrow{p}
Q(u_0)
]
dengan (Q(u_0)) matriks positif definit.

Akibatnya:
[
(X^\top W X)^{-1}
=================

O_p!\big((n h^d)^{-1}\big)
]

---

### 5.6.2 Orde dari (X^\top W^2 X)

Sekarang:

[
X^\top W^2 X
============

\sum_{i=1}^n
w_i^2 X_i X_i^\top
]

Karena:

* (w_i = O((n h^d)^{-1})) secara tipikal (bobot dibagi rata),
* hanya (n h^d) observasi yang relevan,

maka:

[
X^\top W^2 X
============

O_p!\big((n h^d)^{-1}\big)
]

---

### 5.6.3 Gabungkan semuanya

Substitusi orde ke ekspresi variansi:

[
\mathrm{Var}!\big(\hat\beta(u_0)\big)
=====================================

\sigma^2
\underbrace{O_p((n h^d)^{-1})}*{(X^\top W X)^{-1}}
\cdot
\underbrace{O_p((n h^d)^{-1})}*{X^\top W^2 X}
\cdot
\underbrace{O_p((n h^d)^{-1})}_{(X^\top W X)^{-1}}
]

Sehingga:

[
\boxed{
\mathrm{Var}!\big(\hat\beta(u_0)\big)
=====================================

O!\left(\frac{1}{n h^d}\right)
}
]

---

## 5.7 Bentuk limit variansi (lebih presisi)

Dengan normalisasi yang tepat, kita bisa tulis:

[
\mathrm{Var}!\big(\hat\beta(u_0)\big)
\approx
\frac{\sigma^2}{n h^d}
;
Q(u_0)^{-1}
\Omega(u_0)
Q(u_0)^{-1}
]

dengan:

* (Q(u_0) = \mathbb E[X_i X_i^\top \mid U_i \approx u_0])
* (\Omega(u_0) = \mathbb E[w_i^2 X_i X_i^\top \mid U_i \approx u_0])

📌 **Bentuk “sandwich” ini akan muncul lagi di CLT.**

---

## 5.8 Interpretasi (penting, jangan dilewati)

### 5.8.1 Mengapa muncul (n h^d)?

Karena:

* hanya observasi di radius (h) yang dipakai,
* jumlahnya (\asymp n h^d).

Ini adalah **effective local sample size**.

---

### 5.8.2 Trade-off inti GWR

* (h) kecil → bias kecil, tapi variansi besar
* (h) besar → bias besar, tapi variansi kecil

📌 Variansi memberi **harga statistik** dari pelokalan.

---

## 5.9 Ringkasan hasil PART 5

[
\boxed{
\mathrm{Var}!\big(\hat\beta(u_0)\big)
=====================================

O!\left(\frac{1}{n h^d}\right)
}
]

Ini menyiratkan bahwa **skala fluktuasi alami** dari estimator adalah:

[
\sqrt{n h^d}
\big(\hat\beta(u_0) - \beta(u_0)\big)
]

➡️ **Ini tepat skala yang akan kita pakai di CLT.**

---

📌 **STOP DI SINI DULU.**

Bagian berikutnya (**PART 6**) akan:

* menggunakan hasil variansi ini,
* membuktikan **distribusi asimtotik (CLT)**,
* dan menunjukkan **kapan mean-nya nol / tidak nol**.

Jika siap, jawab saja:

> **“lanjut ke distribusi”**

Baik. Kita **lanjut ke PART 6: distribusi asimtotik koefisien lokal**, dan kali ini kita **benar-benar menutup dua pertanyaan inti**:

1. **Apa distribusi limit koefisien lokal?**
2. **Di bawah kondisi apa distribusinya terpusat di nol (valid untuk inferensi)?**

Aku akan **sangat ketat**:
tidak ada istilah baru tanpa penjelasan, tidak ada langkah yang “jelas”, dan setiap normalisasi dijustifikasi.

---

# PART 6

## 6. Penurunan Distribusi Koefisien Lokal

*(GWR dengan kernel hasil GNN, cross-fitted)*

---

## 6.1 Tujuan formal

Kita ingin mempelajari limit distribusi dari estimator:

[
\hat\beta(u_0)
]

secara khusus, kita ingin mencari **skala normalisasi** (a_n) sedemikian sehingga:

[
a_n\big(\hat\beta(u_0) - \beta(u_0)\big)
;\xrightarrow{d};
\text{distribusi non-degenerate}
]

Dari PART 5, kita sudah tahu:

[
\mathrm{Var}\big(\hat\beta(u_0)\big)
====================================

O!\left(\frac{1}{n h^d}\right)
]

➡️ **Ini langsung memberi kandidat normalisasi**:

[
\boxed{
a_n = \sqrt{n h^d}
}
]

---

## 6.2 Kembali ke dekomposisi lengkap (ingat: ini kunci)

Dari PART 4, kita punya:

[
\hat\beta(u_0)
==============

\beta(u_0)
+
B_n(u_0)
+
V_n(u_0)
]

dengan:

* **Bias term**
  [
  B_n(u_0)
  =
  (X^\top W X)^{-1}
  X^\top W X\big(\beta(U)-\beta(u_0)\big)
  ]

* **Noise term**
  [
  V_n(u_0)
  =
  (X^\top W X)^{-1}
  X^\top W \varepsilon
  ]

Kalikan seluruh persamaan dengan (\sqrt{n h^d}):

[
\sqrt{n h^d}\big(\hat\beta(u_0)-\beta(u_0)\big)
===============================================

\underbrace{
\sqrt{n h^d}, B_n(u_0)
}*{\text{bias terstandar}}
+
\underbrace{
\sqrt{n h^d}, V_n(u_0)
}*{\text{noise terstandar}}
]

Distribusi limit **ditentukan oleh dua suku ini**.

---

## 6.3 Distribusi noise term (CLT inti)

Kita fokus dulu ke **noise term**, karena inilah sumber keacakan utama.

### 6.3.1 Tulis noise term sebagai jumlah eksplisit

Ingat:

[
V_n(u_0)
========

(X^\top W X)^{-1}
\sum_{i=1}^n w_i X_i \varepsilon_i
]

Kalikan dengan (\sqrt{n h^d}):

[
\sqrt{n h^d}, V_n(u_0)
======================

\Big(\tfrac{1}{n h^d} X^\top W X\Big)^{-1}
\Big(
\frac{1}{\sqrt{n h^d}}
\sum_{i=1}^n w_i X_i \varepsilon_i
\Big)
]

Perhatikan bahwa ini adalah **produk dua faktor**:

1. satu yang akan konvergen **dalam probabilitas**,
2. satu yang akan konvergen **dalam distribusi**.

---

### 6.3.2 Limit faktor pertama (deterministik secara limit)

Dari PART 5:

[
\frac{1}{n h^d} X^\top W X
;\xrightarrow{p};
Q(u_0)
]

dengan (Q(u_0)) matriks positif definit.

Maka:

[
\Big(\tfrac{1}{n h^d} X^\top W X\Big)^{-1}
;\xrightarrow{p};
Q(u_0)^{-1}
]

---

### 6.3.3 Limit faktor kedua (CLT sesungguhnya)

Sekarang perhatikan:

[
S_n
===

\frac{1}{\sqrt{n h^d}}
\sum_{i=1}^n
w_i X_i \varepsilon_i
]

**Inilah objek yang akan kita CLT-kan.**

---

#### Mengapa ini kandidat CLT?

Karena:

1. Berupa **jumlah dari banyak suku kecil**
2. Tiap suku punya **mean nol**
3. Variansi totalnya **tidak nol dan terbatas**

Kita cek satu-satu.

---

### 6.3.4 Mean nol (di sinilah cross-fitting KRUSIAL)

Karena cross-fitting:

* bobot (w_i) ditentukan dari fold lain,
* sehingga (w_i) **independen dari (\varepsilon_i)** pada fold estimasi.

Maka:

[
\mathbb E[w_i X_i \varepsilon_i \mid X_i, U_i, W] = 0
]

dan:

[
\mathbb E[S_n \mid W] = 0
]

---

### 6.3.5 Variansi terbatas dan stabil

Variansi kondisional:

[
\mathrm{Var}(S_n \mid W)
========================

\frac{1}{n h^d}
\sum_{i=1}^n
w_i^2
\mathbb E[X_i X_i^\top \varepsilon_i^2 \mid U_i \approx u_0]
]

Dari PART 5, jumlah ini konvergen ke:

[
\Omega(u_0)
===========

\mathbb E[w_i^2 X_i X_i^\top \mid U_i \approx u_0]
]

yang **positif definit** dan terbatas.

---

### 6.3.6 Aplikasi CLT

Karena:

* suku-suku independen,
* mean nol,
* variansi terbatas,

maka (CLT Lindeberg–Feller):

[
\boxed{
S_n
;\xrightarrow{d};
\mathcal N\big(0,;\Omega(u_0)\big)
}
]

---

### 6.3.7 Gabungkan dua faktor (Slutsky)

Karena satu faktor konvergen dalam probabilitas dan satu dalam distribusi:

[
\sqrt{n h^d}, V_n(u_0)
;\xrightarrow{d};
\mathcal N\big(
0,;
Q(u_0)^{-1}\Omega(u_0)Q(u_0)^{-1}
\big)
]

---

## 6.4 Peran bias term dalam distribusi

Sekarang kita analisis:

[
\sqrt{n h^d}, B_n(u_0)
]

Dari PART 4, kita tahu:

[
B_n(u_0) = O(h^2)
]

Maka:

[
\sqrt{n h^d}, B_n(u_0)
======================

O\big(\sqrt{n h^d}, h^2\big)
]

---

### 6.4.1 Dua kemungkinan fundamental

#### (i) Bias tidak hilang

Jika:
[
\sqrt{n h^d}, h^2 ;\to; c \neq 0
]

maka:

[
\sqrt{n h^d}\big(\hat\beta(u_0)-\beta(u_0)\big)
;\xrightarrow{d};
\mathcal N\big(
\mu(u_0),;
Q^{-1}\Omega Q^{-1}
\big)
]

dengan mean **tidak nol**:
[
\mu(u_0) \neq 0
]

➡️ **Distribusi normal, tapi tidak terpusat**
➡️ Inferensi Wald standar **tidak valid**

Ini terjadi, misalnya, jika:

* bandwidth dipilih optimal untuk prediksi (CV),
* tidak ada undersmoothing.

---

#### (ii) Bias dihilangkan (kasus inferensi)

Jika kita memaksakan:

[
\boxed{
\sqrt{n h^d}, h^2 \to 0
}
]

yang ekuivalen dengan **undersmoothing**, maka:

[
\sqrt{n h^d}, B_n(u_0) \xrightarrow{p} 0
]

---

## 6.5 Distribusi asimtotik akhir (kasus inferensi sah)

Di bawah kondisi undersmoothing:

[
\boxed{
\sqrt{n h^d}
\big(\hat\beta(u_0)-\beta(u_0)\big)
;\xrightarrow{d};
\mathcal N!\Big(
0,;
Q(u_0)^{-1}\Omega(u_0)Q(u_0)^{-1}
\Big)
}
]

Inilah **jawaban akhir untuk pertanyaan kedua**.

---

## 6.6 Interpretasi (jangan dilewati)

1. **Distribusi normal muncul karena CLT lokal**, bukan karena error normal
2. Skala (\sqrt{n h^d}) = akar dari **effective local sample size**
3. GNN **tidak mengubah bentuk limit**, hanya memengaruhi:

   * matriks (Q(u_0)),
   * matriks (\Omega(u_0))

---

## 6.7 Ringkasan PART 6 (satu napas)

> Setelah dinormalisasi oleh (\sqrt{n h^d}), koefisien lokal GWR dengan kernel yang diestimasi oleh GNN berdistribusi normal asimtotik, asalkan bias dilenyapkan melalui undersmoothing dan bobot dipelajari secara cross-fitted.

---

📌 **STOP DI SINI.**

Yang tersisa untuk menutup seluruh permintaanmu adalah:

* **PART 7** — Ringkasan semua asumsi & desain yang *dipaksa* oleh pembuktian
* **PART 8** — Rangkuman akhir end-to-end:
  *Model → Asumsi → Desain GNN → Estimator → Konsistensi → Variansi → Distribusi*

Jika siap, bilang saja:

> **“lanjut ke ringkasan asumsi”**

Baik. Kita **menutup seluruh rangkaian** dengan **PART 7 dan PART 8**, sesuai format yang kamu minta.
Di sini aku **tidak menambah teori baru**—hanya **merapikan, menegaskan, dan mengikat semuanya secara logis**.

---

# PART 7

## 7. Ringkasan Asumsi-Asumsi dan Desain Model yang Diperlukan

*(disusun sesuai urutan “dipaksa oleh pembuktian”, bukan asal daftar)*

Bagian ini **sangat penting secara akademik**, karena menunjukkan bahwa:

> **asumsi dan desain GNN bukan selera, tapi konsekuensi logis dari inferensi.**

Aku susun dari **paling dasar → paling spesifik**.

---

## 7.1 Asumsi Data dan Model (tidak bergantung pada GNN)

### (A1) Independensi

[
{(Y_i,X_i,U_i)}_{i=1}^n \text{ i.i.d.}
]

➡️ Dipakai untuk:

* LLN,
* CLT lokal.

---

### (A2) Eksogenitas lokal

[
\mathbb E[\varepsilon_i \mid X_i, U_i] = 0
]

➡️ Dipakai untuk:

* mean nol noise term,
* validitas inferensi.

---

### (A3) Variansi error terbatas

[
\mathbb E[\varepsilon_i^2 \mid X_i, U_i] = \sigma^2 < \infty
]

➡️ Dipakai untuk:

* variansi terbatas,
* CLT.

---

### (A4) Momen kovariat terbatas

[
\mathbb E[|X_i|^2] < \infty
]

➡️ Dipakai untuk:

* kontrol (X^\top W X),
* inversibilitas matriks desain lokal.

---

### (A5) Kehalusan fungsi koefisien

[
\beta(\cdot) \in C^2 \text{ di sekitar } u_0
]

➡️ Dipakai untuk:

* Taylor expansion,
* analisis bias (O(h^2)).

---

## 7.2 Asumsi Lokalitas (dipaksa oleh konsistensi)

### (L1) Shrinking neighborhood

[
h \to 0
]

➡️ Dipakai untuk:

* memastikan (\beta(U_i)\approx\beta(u_0)),
* menghilangkan bias.

---

### (L2) Local sample size diverges

[
n h^d \to \infty
]

➡️ Dipakai untuk:

* LLN lokal,
* variansi (\to 0).

---

📌 **Ini menjelaskan mengapa (h) TIDAK bisa dihilangkan**, meskipun pakai GNN.

---

## 7.3 Asumsi Kernel (yang dipenuhi oleh desain GNN)

Ini **tidak ditulis sebagai asumsi kernel klasik**,
melainkan **sebagai syarat yang harus dipenuhi bobot hasil GNN**.

---

### (K1) Non-negativity dan normalisasi

[
w_i(u_0) \ge 0,
\qquad
\sum_i w_i(u_0)=1
]

➡️ Dipakai untuk:

* stabilitas estimator,
* interpretasi sebagai rata-rata lokal.

**Dipenuhi oleh**: softmax pada output GNN.

---

### (K2) Boundedness

[
\sup_i w_i(u_0) \le C (n h^d)^{-1}
]

➡️ Dipakai untuk:

* mencegah satu observasi mendominasi,
* CLT Lindeberg.

**Dipenuhi oleh**: softmax + neighborhood terbatas.

---

### (K3) Regularity (kontinuitas)

[
|w_i(u_0;Z)-w_i(u_0;Z')|
\le
L|Z-Z'|
]

➡️ Dipakai untuk:

* uniform LLN,
* kontrol remainder.

**Dipenuhi oleh**:
arsitektur GNN kontinu (aktivasi halus, tanpa threshold keras).

---

### (K4) Simetri lokal (implisit)

[
\sum_i w_i(u_0)(U_i-u_0) \xrightarrow{p} 0
]

➡️ Dipakai untuk:

* menghilangkan bias orde pertama.

**Dipenuhi oleh desain**:

* input relatif (U_i-u_0),
* graf lokal simetris,
* normalisasi bobot.

---

## 7.4 Asumsi Estimasi Bobot (khusus GNN)

### (G1) Cross-fitting

Bobot (w_i) dipelajari dari data yang **tidak dipakai** untuk estimasi koefisien.

➡️ Dipakai untuk:

* memulihkan eksogenitas,
* memastikan (\mathbb E[w_i\varepsilon_i]=0).

Tanpa ini:

* konsistensi **gagal total**.

---

### (G2) Bounded complexity

Parameter GNN (\theta) berada dalam himpunan kompak.

➡️ Dipakai untuk:

* stabilitas limit,
* mencegah overfitting ekstrem.

---

## 7.5 Asumsi Inferensi (untuk CLT terpusat)

### (I1) Undersmoothing

[
\sqrt{n h^d}, h^2 \to 0
]

➡️ Dipakai untuk:

* menghilangkan bias dalam limit,
* distribusi normal terpusat.

Tanpa ini:

* distribusi tetap normal,
* tapi **mean ≠ 0** (inferensi Wald gagal).

---

# PART 8

## 8. Rangkuman Akhir: Model Formal, Asumsi, dan Desain End-to-End

Bagian ini **menjawab seluruh pertanyaan risetmu secara eksplisit**.

---

## 8.1 Model Formal (ringkas & lengkap)

Data:
[
(Y_i, X_i, U_i), \quad i=1,\dots,n
]

Model:
[
Y_i = X_i^\top \beta(U_i) + \varepsilon_i
]

Target:
[
\beta(u_0)
]

---

## 8.2 Desain Estimator (end-to-end)

### Step 1 — Neighborhood

[
\mathcal N_h(u_0)={i:|U_i-u_0|\le h}
]

---

### Step 2 — Graph lokal

Node:
[
i\in\mathcal N_h(u_0)
]

Fitur node:
[
(X_i,; U_i-u_0)
]

---

### Step 3 — GNN sebagai estimator kernel

GNN menghasilkan skor:
[
s_\theta(i,u_0)
]

Bobot:
[
w_i(u_0)
========

\frac{\exp(s_\theta(i,u_0))}
{\sum_{j\in\mathcal N_h(u_0)}\exp(s_\theta(j,u_0))}
]

---

### Step 4 — Cross-fitting

* GNN dilatih di fold A
* Koefisien dihitung di fold B
* Dirata-ratakan antar fold

---

### Step 5 — Estimasi koefisien lokal

[
\hat\beta(u_0)
==============

(X^\top W_\theta X)^{-1}
X^\top W_\theta Y
]

---

## 8.3 Jawaban eksplisit atas dua pertanyaan utama

### ❓ Apakah koefisien lokal konsisten?

✔️ **YA**, jika:

* neighborhood menyempit,
* fungsi koefisien halus,
* bobot GNN lokal, bounded, dan cross-fitted.

[
\hat\beta(u_0) \xrightarrow{p} \beta(u_0)
]

---

### ❓ Apa distribusi koefisien lokal?

✔️ **Normal asimtotik**, jika additionally:

* undersmoothing dilakukan.

[
\sqrt{n h^d}
(\hat\beta(u_0)-\beta(u_0))
\xrightarrow{d}
\mathcal N!\Big(
0,;
Q(u_0)^{-1}\Omega(u_0)Q(u_0)^{-1}
\Big)
]

Jika tidak:

* distribusi tetap normal,
* tetapi **mean tidak nol**.

---

## 8.4 Ringkasan satu paragraf (siap pakai di tesis)

> *This study establishes the consistency and asymptotic normality of local coefficients in geographically weighted regression with data-adaptive kernels learned via graph neural networks. By preserving shrinking neighborhoods, enforcing kernel regularity through architectural constraints, and employing cross-fitting to restore exogeneity, the proposed estimator achieves valid local inference under standard smoothness and moment conditions.*

---

## Penutup jujur

Apa yang kamu bangun di sini **BUKAN**:

* “GWR + GNN secara heuristik”.

Tapi:

> **kerangka inferensi spasial nonparametrik yang sah secara ekonometrika**,
> dengan GNN sebagai *estimated kernel*.

Kalau kamu mau, langkah berikutnya bisa:

* menurunkan **estimasi variansi yang bisa dihitung di data**,
* menulis **versi LaTeX siap Bab Teori**, atau
* membahas **apa yang gagal jika satu asumsi dihapus** (ini sering ditanya dosen).

Tinggal bilang mau lanjut ke mana.

Ini **pertanyaan kunci yang sangat tepat**, karena di sinilah banyak paper *secara diam-diam* melanggar teori yang baru saja kita bangun.

Aku jawab **tegas tapi adil**, dengan struktur:

1. **Jawaban singkat (ya/tidak + alasan)**
2. **Kriteria matematis: apa yang *harus* dipenuhi backbone GNN**
3. **Backbone mana yang aman / berbahaya (secara konsep, bukan merek)**
4. **Bagaimana membentuk GNN secara praktis agar SELARAS dengan semua asumsi**
5. **Checklist final (praktis & bisa kamu pakai saat desain)**

---

# 1️⃣ Jawaban singkat (jujur)

> ❌ **TIDAK semua backbone GNN bisa digunakan**
> ✔️ **BANYAK backbone populer *harus dimodifikasi***
> ✔️ Yang penting **BUKAN namanya**, tapi **properti matematisnya**

GNN **boleh apa saja** *selama* ia **berperilaku seperti kernel yang sah**
(bounded, smooth, lokal, eksogen secara asimtotik).

---

# 2️⃣ Kriteria matematis yang HARUS dipenuhi GNN

*(ini terjemahan langsung dari pembuktianmu)*

Aku tulis sebagai **syarat → kenapa perlu → implikasi desain**.

---

## (C1) Output harus menghasilkan **bobot kernel**, bukan prediksi

### Syarat

GNN **tidak boleh** memprediksi (Y).

Ia harus menghasilkan **skor bobot**:
[
s_\theta(i,u_0)
\quad\Rightarrow\quad
w_i(u_0)=\text{softmax}(s_\theta)
]

### Kenapa?

Kalau GNN memprediksi (Y):

* bobot jadi fungsi langsung dari noise,
* (\mathbb E[w_i\varepsilon_i]\neq 0),
* **konsistensi gagal total**.

### Implikasi desain

✔️ GNN = *kernel learner*
❌ GNN = *regressor*

---

## (C2) Boundedness & normalization (kernel condition)

### Syarat

[
0 \le w_i \le 1,
\qquad
\sum_i w_i = 1
]

### Kenapa?

* Variansi terbatas
* CLT berlaku
* Tidak ada observasi dominan

### Implikasi desain

✔️ **Softmax WAJIB**
❌ ReLU / linear output langsung ❌

---

## (C3) Kontinuitas / smoothness terhadap input

### Syarat

Bobot harus berubah **halus** saat (U_i) atau (X_i) berubah kecil:
[
|w_i(Z)-w_i(Z')|\le L|Z-Z'|
]

### Kenapa?

* Dipakai di Uniform LLN
* Dipakai di kontrol remainder bias
* Tanpa ini, bukti runtuh diam-diam

### Implikasi desain

✔️ Aktivasi smooth (tanh, GELU, softplus)
❌ Hard attention, threshold, top-k selection

---

## (C4) Locality TIDAK boleh dipelajari oleh GNN

### Syarat

Neighborhood ditentukan **sebelum** GNN:
[
\mathbf 1{|U_i-u_0|\le h}
]

### Kenapa?

* Ini yang menjamin (h\to0)
* Ini yang mendefinisikan (\beta(u_0))

### Implikasi desain

✔️ GNN hanya bekerja *di dalam* neighborhood
❌ GNN memilih node global

---

## (C5) Permutation invariance

### Syarat

Urutan node **tidak memengaruhi output**.

### Kenapa?

* Bobot kernel tidak boleh tergantung indexing
* Ini asumsi implisit semua GWR

### Implikasi desain

✔️ Message passing / aggregation
❌ Sequence-based encoder (tanpa simetrisasi)

---

## (C6) Eksogenitas via cross-fitting

### Syarat

Bobot dihitung dari data yang **tidak digunakan** untuk estimasi koefisien.

### Kenapa?

Tanpa ini:
[
\mathbb E[w_i\varepsilon_i]\neq 0
]

### Implikasi desain

✔️ Train GNN di fold A, apply di fold B
❌ End-to-end training tanpa sample splitting

---

# 3️⃣ Jadi… backbone mana yang bisa / tidak?

⚠️ **Ini bukan daftar hitam-putih**, tapi *status default*.

---

## ✅ Backbone yang *AMAN secara default* (dengan desain tepat)

### ● Message Passing GNN (GCN-like, GraphSAGE-like)

Karena:

* aggregation simetris,
* output kontinu,
* mudah dikontrol.

✔️ Tambahkan softmax → kernel sah

---

### ● Attention-based GNN (GAT-style)

✔️ **BOLEH**, *dengan syarat*:

* attention score → softmax lokal,
* **tidak hard masking**,
* hanya di neighborhood (h).

---

## ⚠️ Backbone yang *BERBAHAYA tanpa modifikasi*

### ● Hard attention / Top-k pooling

Masalah:

* tidak kontinu,
* melanggar (C3),
* bias sulit dikontrol.

---

### ● Deep GNN sangat dalam

Masalah:

* Lipschitz constant bisa meledak,
* ULLN tidak stabil.

Solusi:

* shallow GNN (1–3 layer),
* weight decay / spectral norm.

---

### ● Global Transformer Graph

Masalah:

* tidak lokal,
* kernel tidak shrink,
* target bukan (\beta(u_0)) lagi.

---

# 4️⃣ GNN harus dibentuk seperti apa (arsitektur konkret)?

Aku tuliskan **arsitektur yang “legal”** dari sudut pandang teori.

---

## 🔹 Input

Untuk setiap lokasi target (u_0):

* Node features:
  [
  \mathbf f_i =
  (X_i,; U_i-u_0)
  ]

* Graph:
  [
  \mathcal G_h(u_0)
  ]

---

## 🔹 GNN body (contoh aman)

* 1–2 message passing layer
* Aktivasi smooth
* Tidak ada pooling keras

Output:
[
e_i = \text{GNN}_\theta(\mathbf f_i)
]

*(ini embedding skalar atau vektor kecil)*

---

## 🔹 Kernel head (WAJIB eksplisit)

Mapping embedding → skor:
[
s_i = a^\top e_i
]

Bobot:
[
w_i(u_0)=\frac{\exp(s_i)}{\sum_{j\in\mathcal N_h(u_0)}\exp(s_j)}
]

📌 **Inilah kernel hasil estimasi GNN**

---

## 🔹 Training (cross-fitted)

Loss:
[
\sum_{i\in \text{val}}
\big(Y_i - X_i^\top \hat\beta^{(-i)}(U_i)\big)^2
]

Bukan loss prediksi langsung GNN.

---

# 5️⃣ Checklist praktis (ini emas)

Sebelum pakai backbone GNN, tanyakan:

* ❓ Output GNN jadi **bobot kernel** atau **prediksi Y**?
* ❓ Ada **softmax lokal**?
* ❓ Neighborhood ditentukan **sebelum** GNN?
* ❓ Aktivasi **kontinu**?
* ❓ Ada **cross-fitting**?
* ❓ GNN dangkal & stabil?

Kalau satu saja ❌ → **inferensi tidak sah**.

---

# 6️⃣ Jawaban satu kalimat (siap kamu pakai di diskusi)

> *Not all GNN backbones are suitable for inferential GWR. Only architectures that yield bounded, smooth, locally normalized weights and are trained via cross-fitting preserve the kernel regularity conditions required for consistency and asymptotic normality.*

---

## Penutup jujur

Secara konsep, **GNN di sini bukan “model ML”**, tapi:

> **alat untuk mengestimasi kernel nonparametrik adaptif**.

Kalau kamu mau, langkah lanjut yang sangat kuat:

* **contoh backbone yang SALAH + bukti kegagalannya**, atau
* **pseudo-code arsitektur final siap implementasi**, atau
* **menulis subsection “Neural Kernel Design” versi tesis**.

Tinggal bilang mau yang mana.

Baik. Kita jawab **sangat praktis tapi tetap jujur secara teori**, karena ini titik yang sering disalahpahami.

Aku akan susun jawabannya sebagai **panduan keputusan**, bukan sekadar daftar metode.

---

# Pertanyaan inti yang harus dijawab dulu

Sebelum bicara *cara*, kita luruskan **apa arti “menentukan (h)”** dalam kerangka yang sudah kamu bangun.

> **(h) BUKAN parameter yang harus “optimal” secara numerik.**
> **(h) adalah parameter yang harus “benar secara asimtotik”.**

Artinya:

* kamu **tidak mengejar (h) terbaik di satu dataset**,
* kamu **mengejar aturan pemilihan (h_n)** yang:
  [
  h_n \to 0
  \quad\text{dan}\quad
  n h_n^d \to \infty
  ]

Inferensi **hidup–mati** di sini.

---

# Ringkasan jawaban (biar orientasi)

> **Dalam praktik inferensial (seperti modelmu):**
>
> ✔️ Jangan pilih (h) murni dari CV
> ✔️ Tentukan (h) dengan **aturan deterministik atau kNN**
> ✔️ Gunakan CV hanya sebagai **kalibrasi kasar**, bukan penentu akhir
> ✔️ Biarkan GNN mengurus adaptivitas, bukan (h)

Sekarang kita rinci satu per satu.

---

# OPSI 1 — Deterministic bandwidth (PALING AMAN UNTUK TESIS)

### Bentuk

[
\boxed{
h_n = C \cdot n^{-\alpha}
}
]

dengan:

* (C>0) konstanta skala,
* (\alpha>0) laju penyusutan.

---

## Kenapa ini paling aman?

Karena kamu **secara eksplisit** bisa menyatakan:

* (h_n \to 0) (karena (\alpha>0)),
* (n h_n^d \to \infty) (asal (\alpha < 1/d)).

Untuk **inferensi GWR**, kita butuh **undersmoothing**:
[
\sqrt{n h_n^d}, h_n^2 \to 0
]

Untuk data spasial 2D ((d=2)), pilihan aman:
[
\alpha \in (0.25,;0.4)
]

Contoh konkret:

* (n=1{,}000) → (h \approx C\cdot 1000^{-0.3})
* (n=10{,}000) → (h) lebih kecil otomatis

📌 **Ini pilihan terbaik jika kamu ingin dosen tidak bisa “menyerang” secara teori.**

---

## Bagaimana memilih (C)?

Ini **bukan masalah teori**, tapi skala data.

Praktik aman:

* hitung jarak median antar lokasi,
* set (C) sebagai proporsi jarak itu (mis. 0.5×).

Yang penting:

> (C) **tetap**, tidak tergantung (n).

---

# OPSI 2 — k-Nearest Neighbors (SANGAT NATURAL UNTUK GNN)

Alih-alih radius (h), kamu pilih **jumlah tetangga**.

---

### Bentuk

[
\boxed{
k_n = \lceil n^\gamma \rceil,
\quad 0<\gamma<1
}
]

Neighborhood:

* ambil (k_n) tetangga terdekat dari (u_0),
* (h_n(u_0)) = jarak ke tetangga ke-(k_n).

---

## Kenapa ini sah secara teori?

Karena:
[
h_n(u_0)
\asymp
\left(\frac{k_n}{n}\right)^{1/d}
]

Jika:

* (k_n \to \infty),
* (k_n/n \to 0),

maka:
[
h_n \to 0
\quad\text{dan}\quad
n h_n^d \asymp k_n \to \infty
]

📌 **Semua syarat inferensi terpenuhi.**

---

## Pilihan (\gamma) yang aman (praktis)

Untuk (d=2):

* (\gamma \in (0.4,;0.7))

Contoh:

* (n=2{,}000) → (k\approx 60)
* (n=10{,}000) → (k\approx 150)

📌 Ini **sangat cocok** dengan graf GNN:

* graf lokal = kNN graph,
* tidak perlu radius eksplisit.

---

# OPSI 3 — Cross-validation (HANYA SEBAGAI REFERENSI)

Ini penting, karena kamu pasti ditanya.

---

### Fakta jujur tentang CV

CV memilih:
[
h_{CV} \asymp n^{-1/(d+4)}
]

Untuk (d=2):
[
h_{CV} \asymp n^{-1/6}
]

Masalah:
[
\sqrt{n h_{CV}^d}, h_{CV}^2 \to c \neq 0
]

➡️ Bias **tidak hilang** dalam CLT
➡️ Inferensi Wald **tidak sah**

---

## Cara CV masih boleh dipakai

Sebagai **starting point**, bukan keputusan akhir.

Praktik yang jujur:

1. Hitung (h_{CV})

2. Lakukan **undersmoothing**:
   [
   \boxed{
   h = c \cdot h_{CV},
   \quad c \in (0.5,;0.8)
   }
   ]

3. Jelaskan di teks:

> CV digunakan untuk kalibrasi skala, bukan untuk inferensi.

---

# Apa yang TIDAK boleh dilakukan

❌ Memilih (h) yang meminimalkan MSE dan berhenti di situ
❌ Mengklaim “(h) kecil” tanpa aturan asimtotik
❌ Membiarkan GNN menentukan neighborhood
❌ Menggunakan (k) tetap (mis. 50) tanpa kaitan ke (n)

---

# Diagnostic PRAKTIS (bukan bukti, tapi sanity check)

Ini **sangat disukai penguji**.

### 1️⃣ Effective local sample size

Pastikan:
[
30 \le n h^d \le 0.2n
]

Kalau:

* terlalu kecil → variance meledak,
* terlalu besar → tidak lokal.

---

### 2️⃣ Stabilitas koefisien

Hitung (\hat\beta(u_0)) untuk:
[
h,; 0.8h,; 0.6h
]

Koefisien **harus stabil**, bukan meloncat.

---

# Rekomendasi AKHIR untuk modelmu (tegas)

Karena kamu:

* ingin inferensi,
* memakai GNN sebagai kernel adaptif,

maka **pilihan terbaik**:

> ✔️ **kNN-based bandwidth dengan (k_n \propto n^\gamma)**
> ✔️ atau deterministic (h_n = C n^{-\alpha})
> ❌ CV murni

Kalimat tesis yang kuat:

> *“The bandwidth is chosen to guarantee shrinking neighborhoods and diverging local sample size, while adaptivity is achieved through learned neural weights rather than bandwidth optimization.”*

---

# Ringkasan satu baris (hafalkan)

> **Dalam inferensi GWR–GNN, (h) dipilih untuk memenuhi teori, bukan untuk memaksimalkan prediksi.**

Kalau kamu mau, langkah berikutnya yang sangat natural:

* menulis **subsection “Bandwidth Selection” versi skripsi**, atau
* membahas **apa yang terjadi jika (h) salah dipilih (counterexample)**, atau
* mengaitkan **pemilihan (h) dengan desain graf GNN secara eksplisit**.

Tinggal bilang mau yang mana.
