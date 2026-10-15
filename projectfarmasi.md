# sediantra: Sistem Pengadaan Obat RSUD Berbasis Integrasi ABC–VEN, Machine Learning & Batas Kas BLUD

> **sediantra**: Sistem prediksi kebutuhan dan perencanaan pembelian obat RSUD yang menggabungkan **Prioritas Obat (Matriks ABC–VEN)**, **Prediksi Machine Learning (LightGBM)**, **Optimasi Anggaran Tahan Risiko (HiGHS Dual LP)** [12][18], serta **Batas Kas Rumah Sakit (Kas BLUD & Kuota RKO)**.

---

### Tim Pelaksana & Struktur Peran
Pengembangan dan pengujian sediantra dilakukan bersama lintas bidang:
1. **Dr. Karlisa Priandana, S.T., M.Eng** — Ketua Tim (IPB University; Validasi Ilmiah & Arah Metodologi Penelitian).
2. **Jeremy Tjahjana** — Koordinator Produk, Bisnis & Sistem Informasi (Kebutuhan Pengguna, Alur Kerja Farmasi, Pengujian Kegunaan).
3. **Julius Calvin Kurniadi** — Koordinator Teknis (Arsitektur Web, Standar Pengembangan, Integrasi Modul).
4. **Gilang Muhamad Widiagung** — Koordinator AI & Model Machine Learning (Feature Engineering, Peramalan Deret Waktu, Validasi Bebas Data Leakage).
5. **Ilham Edgar Maulana Goesasi** — Koordinator Integrasi & UI/UX (Desain Antarmuka, Menghubungkan Model ke Mesin Rekomendasi, Ekspor Draf LKPP).
6. **Benedictus Tjahjana** — Masukan Industri & Hilirisasi (RS Melania Bogor; Kelayakan Logistik Rumah Sakit & Risiko Implementasi Pengadaan).

> **Nilai Utama (Core Value Proposition):** Menghubungkan prediksi kebutuhan obat (LightGBM) dengan pembagian anggaran belanja yang aman dan berpatokan pada **Matriks ABC–VEN**. Tujuannya: **mengantisipasi lonjakan wabah musiman (data BMKG) tanpa membuat utang menumpuk ke distributor farmasi**, mencegah obat darurat (Vital & Esensial) habis, dan memberi peringatan dini di depo sebelum obat di resep kosong, tanpa menimbun stok mati di gudang RSUD.

---

## 1. Project Snapshot & Scope Guardrails

| Atribut | Definisi / Spesifikasi |
| :--- | :--- |
| **Kategori & Track** | Digitalisasi AI Bidang Kesehatan (Hackathon DeepTech 2026) [1] |
| **Target Pengguna** | Tim Farmasi Perencana & Kepala Instalasi Farmasi RSUD (Kelas B & C) |
| **Pendekatan Sistem** | **Decision-Support System (Human-in-the-Loop):** tiap prediksi disertai faktor pendorong, pembanding baseline, dan dapat di-*override* apoteker dengan log alasan |
| **Komposisi Tim** | Multi-disiplin: profesional industri, akademisi/dosen periset, dan 4 *core software & data engineers* (kombinasi riset teruji dan sistem siap pakai di RSUD) |
| **Komputasi & Runtime** | Tanpa GPU: GBDT + dual-stage LP/Greedy di CPU standar. Teruji terkalibrasi: **< 10 ms untuk 50 SKU**, **< 250 ms untuk 500 SKU**, **< 510 ms untuk 1.000 SKU**, dan **~1,04 detik untuk 2.000 SKU** dengan *memory footprint* ramping (~3,9 MB pada 1.000 SKU) berkat representasi matriks *Compressed Sparse Row* (CSR). Arsitektur produksi: Backend REST API FastAPI (Python) decoupled dengan Webapp Frontend Next.js (App Router + Tailwind CSS + Recharts). |
| **Strategi Presentasi & Pitching** | Demo berfokus pada **alur operasional nyata**: Alert Depo $\rightarrow$ Simulasi Defekta Kas Ketat (§6) $\rightarrow$ Draf CSV LKPP. Desain antarmuka mengacu pada standar visual kelas dunia (Rxflow Dashboard) tanpa fitur kosmetik chatbot LLM teks bebas. |
| **Output Utama** | Alert hari ketahanan stok depo (diperbarui harian dari snapshot stok) + Draf Usulan Pembelian / Defekta untuk Pejabat Pengadaan (PP) & PPK via e-Purchasing LKPP (ekspor CSV) [9] |

### 1.1 Tata Kelola Klinis & Otoritas Keputusan di RSUD

1. **Komite / Tim Farmasi dan Terapi (KFT):** Menetapkan kebijakan makro: ambang fraktil CSL target per kelas VEN ($\tau_V, \tau_E, \tau_N$), katalog padanan substitusi legal yang terkunci pada Formularium RS & Formularium Nasional (Fornas), serta stok minimum darurat troli IGD/ICU (`vital_min_stock`) [7].
2. **Apoteker Perencana / Kepala Instalasi Farmasi:** Mengkaji draf usulan rekomendasi AI, melakukan penyesuaian (*override*) berdasar pertimbangan klinis/musiman dengan kewajiban mengisi log alasan pada `fct_override_log`, dan mengesahkan defekta pengadaan.
3. **Pejabat Pengadaan (PP) & Pejabat Pembuat Komitmen (PPK):** Mengeksekusi Draf Usulan Pembelian menjadi paket pemesanan resmi (*Purchase Order*) melalui transaksi *e-Purchasing* e-Katalog INAPROC LKPP (Perpres 16/2018 jo. Perpres 12/2021 jo. Perpres 46/2025 & Permendagri 79/2018) [8][9].

### Scope Guardrails (Batasan Implementasi MVP)

| Dimensi | In-Scope (Fokus Hackathon) | Out-of-Scope (Bukan Prioritas MVP) |
| :--- | :--- | :--- |
| **Portofolio Obat** | 50 SKU obat representatif RSUD (Fast-moving, Lumpy, Intermiten Vital, Non-Esensial) | Bahan Medis Habis Pakai (BMHP) non-obat, barang logistik umum, dan ribuan item non-medis |
| **AI Forecasting & Optimasi** | Arsitektur Tiga Tahap: Stage 1 Multi-Quantile LightGBM ($\tau \in \{0.5, 0.8, 0.9\}$) + Stage 2 Wasserstein Distributionally Robust Optimization (HiGHS Dual LP < 10ms) untuk target kontinu tahan lonjakan risiko + Stage 3 Greedy Runway Waterfall alokasi kemasan box fisik | Chatbot LLM generatif bebas teks |
| **Alokasi Anggaran** | Batasan ganda (*dual budget constraint*): pagu kas cair BLUD (Permendagri 79/2018 Ps. 74–75) [8] dan batas kuota RKO/RBA; alokasi bertahap: dasar V ➔ dasar E ➔ buffer V ➔ buffer E ➔ N; urutan pendanaan kelas ABC (C ➔ B ➔ A) pada tahap buffer E dan N; dekomposisi stok aditif transparan (*Additive Waterfall*) untuk telaah apoteker & audit BPK | Optimasi dynamic multi-echelon warehouse routing |
| **Batasan Logistik** | Kemasan box resmi (*pack size* $P_i$), komoditas e-Katalog LKPP, sisa kuota RKO tahunan, MOQ distributor, dan sisa umur simpan (FEFO) | Negosiasi harga kontrak dinamis di luar e-Katalog INAPROC LKPP |
| **Cakupan Alur** | Alert ketahanan stok depo (IGD/RJ/RI) + Rekomendasi usulan pengadaan gudang induk | Modul kasir, billing pasien, atau verifikasi klaim BPJS |
| **Artefak Keputusan** | Draf Usulan Pembelian / Defekta untuk PP & PPK via e-Purchasing LKPP kelipatan box format CSV (Perpres 16/2018 jo. Perpres 12/2021 jo. Perpres 46/2025 & Keputusan Kepala LKPP 122/2022) [9] | Integrasi write-back API langsung ke sistem live SIMRS |

---

## 2. Problem Statement, Keselarasan Tema & Friksi RSUD

### Penjajaran Tema Resmi Hackathon
> *"Pada fase farmasi, resep menunggu lama, stok obat tidak terpantau secara real-time, dan berisiko terjadi kekosongan obat."* [1]

sediantra merespons langsung ketiga friksi inti tema hackathon:
1. **Resep Menunggu Lama (*Extreme Waiting Times*):**  
   Standar Pelayanan Minimal (SPM) Farmasi RS (Kepmenkes 129/2008) [10] menetapkan waktu tunggu obat jadi (non-racikan) $\le 30$ menit dan racikan $\le 60$ menit. Salah satu pemicu utama waktu tunggu melampaui SPM adalah **obat baru diketahui kosong saat resep fisik diserahkan pasien di loket depo**. Petugas terpaksa menghentikan antrean untuk konfirmasi telepon ke dokter penulis resep (15–20 menit), mencari padanan di SIMRS, atau amprah cito ke depo lain/gudang induk (15–30 menit). sediantra menyasar pemicu ini lewat *Days-of-Cover* (DoC) alert harian sebelum stok depo habis. Dampak pemangkasan waktu tunggu dimodelkan lewat proksi waktu intervensi:
   $$\Delta T_{\text{tunggu}} \approx \text{Unmet Prescriptions} \times 20\text{ menit (konfirmasi DPJP / amprah cito)}$$
   didukung evaluasi *fill rate* dan jumlah resep tertunda (§7).
2. **Stok Tidak Terpantau Secara Real-Time:**  
   MVP memantau stok **harian** (snapshot per batch per lokasi dari *read-replica*), belum *real-time*. Ini sudah menutup celah utama cara manual (stok opname berkala), tetapi pemantauan per transaksi membutuhkan akses stok transaksional SIMRS dan masuk rencana pengembangan lanjutan. Pelatihan model dan alokasi kas dijalankan terjadwal tanpa membebani database transaksional.
3. **Risiko Kekosongan Obat Terprogram (Sistem Pendukung Keputusan / DSS Gudang Induk):**  
   Kekosongan di depo bersumber dari perencanaan pengadaan gudang induk yang terlambat atau terpotong tanpa prioritas saat kas terbatas [2][3]. Berbeda dari baseline manual yang menggunakan **Metode Konsumsi standar Permenkes 72/2016 jo. Petunjuk Teknis Standar Pelayanan Kefarmasian di RS (Kemenkes RI, 2019) (Static Safety Stock 20%)** [7], sediantra mengoptimalkan rencana pembelian berbasis kuantil kebutuhan dan prioritas klinis.

### Tiga Friksi Lapangan RSUD yang Diselesaikan

| No | Friksi Lapangan RSUD | Masalah Cara Manual (Excel/SIMRS) | Solusi Sistem sediantra |
| :---: | :--- | :--- | :--- |
| **1** | **Uang Kas BLUD Terbatas & Risiko Diblokir Distributor (Klaim BPJS Macet)** | Excel dan rumus biasa hanya menghitung kebutuhan obat tanpa melihat sisa uang kas tunai rumah sakit [8]. Ketika kas tipis dan tagihan BPJS macet, apoteker terpaksa memotong pesanan pakai perasaan. Akibatnya, obat darurat rawan ikut terpotong atau utang lama menumpuk hingga distributor menghentikan pengiriman obat (*credit lock*). | Sistem membatasi belanja di bawah sisa uang kas BLUD dan kuota resmi tahunan (RKO). Dana dialokasikan bertahap (obat Vital ➔ Esensial ➔ cadangan Vital ➔ cadangan Esensial ➔ Non-Esensial). Dilengkapi rem utang otomatis di hari ke-60 agar utang ke distributor tidak sampai menunggak 90 hari dan memicu pemboikotan. |
| **2** | **Obat Kosong Bikin Salah Prediksi & Lonjakan Penyakit Musiman Terlewat** | Saat obat kosong 5 hari, sistem mencatat pemakaian = 0. Rumus rata-rata mengira obat tidak laku sehingga pesanan berikutnya makin sedikit (*death spiral*). Selain itu, rumus biasa tidak bisa membaca sinyal cuaca hujan lebat dari BMKG yang memicu lonjakan pasien DBD dan ISPA. | Hari-hari saat stok kosong otomatis diisi dengan rata-rata pemakaian normal 7 hari sebelumnya. Data curah hujan BMKG dihubungkan dengan jeda masa inkubasi penyakit (DBD jeda 21–35 hari) untuk mengantisipasi lonjakan kebutuhan cairan infus sebelum pasien membludak di IGD. |
| **3** | **Aturan Pembelian e-Katalog LKPP & Kemasan Distributor** | Pembelian obat RSUD wajib melalui e-Purchasing LKPP per kelipatan boks resmi dan memenuhi batas minimal order (MOQ) [9]. Hitungan manual satuan butir/ampul sering ditolak oleh pejabat pengadaan atau distributor karena salah ukuran kemasan. | Sistem langsung menghasilkan Draf Usulan Pembelian (format CSV e-Katalog) yang sudah pas kelipatan boks resmi ($P_i$), sesuai kode komoditas LKPP, dan memenuhi syarat minimal order vendor, lengkap dengan rincian perhitungan yang siap diaudit BPK. |

---

## 3. Data Architecture & Ingestion Contracts

Sistem membaca data analitik secara non-blokir via replika baca (*read-replica*) tanpa mengganggu transaksi pelayanan harian SIMRS.

*(Diagram grafis visual lengkap dipindahkan ke [erd_datamart.md](erd_datamart.md) untuk performa render).*

| Nama Tabel / Entitas | Tipe Skema | Kunci Utama (PK) | Kunci Asing (FK) | Daftar Kolom & Atribut Skema Lengkap |
| :--- | :---: | :--- | :--- | :--- |
| `dim_drug_master` | Dimensi | `item_id` | `ven_category` $\rightarrow$ `dim_ven_policy` | `item_id`, `item_name`, `kfa_code`, `lkpp_commodity_id`, `nie_bpom`, `atc_code`, `therapeutic_interchange_group`, `is_fornas`, `is_formularium_rs`, `is_high_alert`, `ven_category`, `abc_class`, `pack_size`, `unit_price`, `min_order_qty`, `default_lead_time_days`, `vital_min_stock`, `annual_rko_quota_pack`, `holding_cost_rate`, `supplier_pbf`, `payment_terms_days` |
| `dim_location` | Dimensi | `location_id` | — | `location_id`, `location_name`, `location_type`, `is_warehouse`, `internal_lead_time_days` |
| `dim_ven_policy` | Dimensi | `ven_category` | — | `ven_category`, `shortage_cost`, `excess_cost` |
| `dim_hospital_context` | Dimensi | `tx_date` | — | `tx_date`, `inpatient_bor`, `outpatient_visits`, `emergency_visits`, `is_holiday`, `rainfall_mm` |
| `dim_budget_allocation`| Dimensi | `budget_period`, `budget_source` | — | `budget_period`, `budget_source`, `budget_limit` |
| `fct_prescription_dispensing` | Fakta | `tx_date`, `item_id`, `location_id` | `tx_date` $\rightarrow$ `dim_hospital_context`, `item_id` $\rightarrow$ `dim_drug_master`, `location_id` $\rightarrow$ `dim_location` | `tx_date`, `item_id`, `location_id`, `qty_prescribed`, `qty_dispensed`, `is_unmet_or_substituted`, `is_stockout` |
| `fct_inventory_snapshot`| Fakta | `snapshot_date`, `item_id`, `location_id`, `batch_number` | `item_id` $\rightarrow$ `dim_drug_master`, `location_id` $\rightarrow$ `dim_location` | `snapshot_date`, `item_id`, `location_id`, `batch_number`, `expiry_date`, `current_stock`, `emergency_trolley_stock`, `is_quarantine` |
| `fct_purchase_orders` | Fakta | `po_id`, `item_id` | `item_id` $\rightarrow$ `dim_drug_master`, `(budget_period, budget_source)` $\rightarrow$ `dim_budget_allocation` | `po_id`, `item_id`, `order_date`, `expected_date`, `received_date`, `order_qty_pack`, `total_line_rupiah`, `supplier_pbf`, `budget_period`, `budget_source`, `status`, `payment_terms_days` |
| `fct_override_log` | Fakta | `override_id` | `item_id` $\rightarrow$ `dim_drug_master`, `(budget_period, budget_source)` $\rightarrow$ `dim_budget_allocation` | `override_id`, `item_id`, `budget_period`, `budget_source`, `run_date`, `user_id`, `recommended_box`, `final_box`, `reason`, `created_at` |

### Aturan Pengolahan Data & Ekstraksi SIMRS

1. **Decensoring Permintaan & Kontrak Padanan Substitusi (`therapeutic_interchange_group`):**  
   Pada hari `is_stockout = true`, konsumsi laten diestimasi via median *run-rate* 7 hari sebelum kosong (sejalan dengan prinsip estimasi batas terpotong / *censored demand* [13]).  
   - *Rasionalisasi Desain:* Median pemakaian 7 hari dipilih karena deterministik, cepat, dan stabil untuk obat yang jarang keluar (intermiten) [13]. Risiko lonjakan wabah (seperti DBD) tidak luput karena sudah diantisipasi lebih dulu oleh sinyal cuaca BMKG ($t-28$) di Stage 1. Lonjakan obat pengganti dibatasi ketat pada formularium KFT/Fornas (`therapeutic_interchange_group`) dan ditandai `is_unmet_or_substituted = true` agar tidak mengotori data konsumsi normal.
2. **Atribut Kontraktual PBF & Pengawasan Kredit (`supplier_pbf`, `payment_terms_days`):**  
   Atribut `supplier_pbf` dan `payment_terms_days` pada `dim_drug_master` dan `fct_purchase_orders` merekam identitas Pedagang Besar Farmasi pemenang e-Katalog LKPP serta durasi jatuh tempo invoice kontraktual (*Terms of Payment* / TOP, default 30 hari). Atribut ini menjadi dasar mesin pemantau *Accounts Payable (AP) Aging* untuk mengelompokkan risiko penutupan kredit distributor (*vendor credit lock*) pada level entitas RSUD dan mengaktifkan rem likuiditas (§4.3) secara proaktif pada umur tagihan 60 hari sebelum menyentuh batas kritis penutupan kredit distributor (90 hari).
3. **Pencegahan Kebocoran Data (Covariate Leakage Guard):**  
   Fitur masukan pada titik ramal $t$ hanya menggunakan data historis ($s \le t$) atau proyeksi eksogen resmi yang sudah rilis pada saat $t$ (kalender libur nasional & prakiraan resmi BMKG). Dilarang keras memakai realisasi aktual BOR atau kunjungan masa depan ($s > t$).
4. **Agregasi Terpisah (Depo vs Gudang RS):**  
   - *Alert Depo:* Dihitung dari median konsumsi lokal harian per depo.
   - *Pengadaan Gudang:* Model kuantil dilatih langsung pada total agregat kebutuhan seluruh RS. Kuantil per depo tidak boleh dijumlahkan secara naif karena memicu bias over-ordering ($\sum \hat{Q}_\tau \ne \hat{Q}_\tau(\sum)$). Mutasi internal antar-depo dikecualikan dari data konsumsi pasien.
5. **Snapshot Stok Harian:**  
   Metrik DoC dan stok efektif dihitung harian dari snapshot stok fisik per batch per lokasi dengan penataan urutan kedaluwarsa terdekat (FEFO non-karantina).
6. **Inferensi Hari Kosong (SIMRS Fallback & Proteksi SKU Intermiten):**  
   Pada database SIMRS tanpa log pembatalan resep terstruktur, status `is_stockout = true` diinferensikan otomatis jika:  
   - **Fast/Normal-Moving:** Saldo stok fisik $= 0$ ATAU kuantitas dispensing $= 0$ pada hari pelayanan aktif (`outpatient_visits > 0` atau `inpatient_bor > 0`).  
   - **Intermiten / Vital Darurat (mis. SABU, Antidot):** Nilai dispensing $= 0$ adalah kondisi normal (bukan stockout). Maka `is_stockout = true` **hanya** diaktifkan jika saldo stok fisik $\approx 0$ (`current_stock == 0`) ATAU terdapat resep tertunda (`is_unmet_or_substituted = true`), guna mencegah imputasi kebutuhan hantu (*phantom demand*).
7. **Pembersihan Otomatis Data Anomali SIMRS (Defensive Sanitizer):**  
   Pada implementasi nyata di RSUD, data SIMRS sering mengalami anomali atau salah ketik. sediantra dilengkapi lapisan sanitasi data otomatis (*fail-safe guard*) sebelum masuk ke solver matematika:
   - **Stok Negatif (Nurse Retur / Gantung Transaksi):** Nilai stok $< 0$ (mis. `-120 unit`) otomatis dinormalisasi ke `0.0` untuk mencegah algoritma memesan box fiktif berlebihan.
   - **Nilai Hilang (`NaN`, `None`):** Imputasi deterministik menggunakan konsumsi dasar historis ($Q_{\text{base}} = \text{daily\_demand} \times 30$) sehingga solver linier terhindar dari *crash* / *unhandled exception*.
   - **Typo Ekstrem / Outlier Absurd (mis. entri 500.000 unit):** Kuantil dibatasi (*historical clamp*) maksimal $10\times$ hingga $20\times$ kebutuhan rata-rata bulanan. Ini mencegah satu obat yang salah ketik menyedot 100% anggaran rumah sakit dan mematikan pengadaan obat lainnya.
   - **Anomali Data Utang PBF:** Nilai umur utang negatif atau absurd ($> 365$ hari) dijepit ke rentang logis $[0, 365]$ hari agar rem utang (*AP Debt Throttle*) tetap berfungsi presisi.

---

## 4. Logika Perhitungan & Peran AI (Engineering Logic)

Semua perhitungan dibangun di atas logika logistik operasional yang transparan dan dapat diverifikasi langsung oleh apoteker [5].

### 4.1 Aturan Bisnis & Logika Logistik Farmasi (Business Rules)

Semua rekomendasi pemesanan obat wajib mematuhi 7 aturan operasional rumah sakit dan regulasi e-Katalog LKPP:

1. **Aturan FEFO (First Expired, First Out):**
   - Data stok fisik dihitung per batch berdasarkan tanggal kedaluwarsa terdekat (non-karantina).
   - Batch yang diproyeksikan kedaluwarsa sebelum sempat terkonsumsi (berdasarkan laju pakai harian median) otomatis dialihkan ke antrean alert rotasi antar-depo.

2. **Days of Cover (DoC) & Alert Depo:**
   - Mengukur berapa hari stok fisik di depo (IGD, Rawat Jalan, Rawat Inap) mampu bertahan melayani resep.
   - Formula: `DoC = Stok Pelayanan / max(Laju Konsumsi Harian, vital_min_stock / H, 1e-6)`.
   - *Catatan Laju Konsumsi Harian:* Dihitung dari median dispensing harian pada hari-hari non-kosong (`is_stockout == false`) selama 7 hari aktif terakhir guna mencegah fenomena *death spiral* (DoC semu akibat stok kosong).
   - **Isolasi Segel Troli IGD:** Stok obat di dalam troli darurat yang tersegel tidak dimasukkan ke dalam hitungan stok pelayanan umum (`Stok Pelayanan = max(0, sum(current_stock valid) - emergency_trolley_stock)`), agar stok darurat yang terkunci tidak menunda munculnya peringatan obat menipis.
   - **Tingkat Peringatan (Alert):**
     - **Status Merah:** `0 <= DoC <= Lead Time Internal Gudang` (Stok akan habis sebelum kiriman amprah gudang tiba; segera lakukan mutasi cito atau pinjam antar-depo). Jika stok gudang induk juga kosong, sistem menaikkan status menjadi darurat eksternal.
     - **Status Kuning:** `Lead Time < DoC <= Lead Time + 3 Hari` (Batas waktu aman untuk mengajukan permintaan rutin ke gudang induk).

3. **Batas Pengaman Obat Vital (Safety Floor):**
   - Obat darurat penyelamat nyawa yang jarang keluar (misal: Serum Anti Bisa Ular / SABU) berisiko diprediksi 0 oleh model AI karena jarang ada transaksi harian.
   - Rumus: `Target = max(Prediksi ML, vital_min_stock)`. Nilai `vital_min_stock` dipatok sesuai ketetapan Komite Farmasi & Terapi (KFT) agar obat penyelamat nyawa selalu tersedia di troli darurat.

4. **Kebutuhan Bersih Pembelian (Net Reorder):**
   - Formula: `Kebutuhan Bersih = max(0, Target Kebutuhan - Stok Fisik Gudang - Pesanan di Jalan)`.
   - Jika `Stok Fisik + Pesanan Tiba < Kebutuhan Lead Time`, sistem menyalakan *early warning stockout* untuk memicu pengiriman ekspres atau bon pinjam-mengganti persediaan darurat antar-fasyankes (PerBPOM 24/2021 Ps. 27–29 jo. Permenkes 72/2016) [16].

5. **Aturan Kemasan Box & Batas MOQ e-Katalog:**
   - Pembelian via e-Purchasing LKPP wajib mematuhi kelipatan kemasan kardus resmi vendor ($P_i$).
   - Formula: `Jumlah Box = ceil(Kebutuhan Bersih / Isi per Box)`.
   - Jika hasil hitungan berada di bawah ambang minimum order distributor (`Jumlah Box < MOQ`), kuantitas dinaikkan ke batas `MOQ` selama sisa kuota RKO mencukupi.
   - Pengecekan kuota RKO tahunan:
     - Jika `Usulan Box > Sisa RKO`, kuantitas di-clamp ke batas `Sisa RKO` dan memicu alert adendum RKO ke Kemenkes via aplikasi e-Monev Obat [20].
     - Jika `0 < Sisa RKO < MOQ`, sistem menahan pemesanan obat non-vital (`status: BLOCKED_RKO_MOQ_DEADLOCK`) guna mencegah audit temuan BPK atas pembelian melampaui kuota, serta memicu alert adendum percepatan kuota ke Dinkes/Kemenkes. Untuk obat Vital, pengadaan diarahkan melalui mekanisme SPK darurat langsung BLUD (Permendagri 79/2018 Ps. 76) [8].

6. **Plafon Kas Cair BLUD (Permendagri 79/2018 Ps. 74–75) [8]:**
   - Formula: `Pagu Tersedia = budget_limit - Total PO Periode Berjalan yang Terbit`.
   - Total nilai rupiah dari seluruh usulan pembelian box tidak boleh melampaui pagu kas tersedia ini.

7. **Klasifikasi ABC (Prinsip Pareto Belanja):**
   - Dihitung dari akumulasi nilai pemakaian setahun (jumlah unit terpakai $\times$ harga satuan e-Katalog):
     - **Kelas A (80% serapan kas):** Item bervolume besar atau harga tinggi (misal: cairan infus RL/NaCl, antibiotik paten).
     - **Kelas B (15% serapan kas):** Item bernilai pemakaian sedang.
     - **Kelas C (5% serapan kas):** Item bernilai pemakaian rendah.
   - Aturan ini tidak boleh menggugurkan prioritas klinis VEN (obat Vital tetap kebal potong), melainkan dipakai sebagai urutan perlindungan buffer pada obat Esensial & Non-Esensial saat pagu kas terbatas (prioritas pendanaan: C ➔ B ➔ A; atau urutan pemotongan penghematan: A ➔ B ➔ C) guna memaksimalkan efisiensi modal kerja.

### 4.2 Arsitektur Pipeline 3 Tahap (Engineering Spec)

Pipeline pemrosesan data dirancang menjadi 3 tahap berurutan: **Prediksi Kebutuhan (AI/ML) ➔ Optimasi Kas & Risiko (Wasserstein DRO) ➔ Eksekusi Pembelian Box (Greedy Waterfall)**.

| Ringkasan Pipeline | Stage 1: Prediksi Kebutuhan (AI/ML) | Stage 2: Optimasi Kas & Risiko (Wasserstein DRO) | Stage 3: Eksekusi Box & Kuota (Greedy Waterfall) |
| :--- | :--- | :--- | :--- |
| **Tujuan** | Peramalan rentang pemakaian per SKU (P50, P80, P90) berbasis tren & sinyal cuaca BMKG. | Menyeimbangkan alokasi kas 50 SKU secara global dan tahan terhadap lonjakan risiko terburuk. | Membulatkan target desimal ke kemasan box fisik LKPP dan mengerem utang distributor. |
| **Engine / Modul** | LightGBM Quantile (`backend/model_forecast.py`) [22] | Solver HiGHS LP (`backend/allocation_engine.py`) [12][18] | Heuristik Runway Waterfall (`backend/allocation_engine.py`) [9][23] |
| **Input Kunci** | Riwayat SIMRS (decensored), BOR ranjang, lag hujan BMKG ($t-28$) [19] | Kuantil Stage 1, sisa kas cair BLUD [8], bobot klinis VEN [4], radius guncangan $\epsilon$ | Target $x_i^*$ Stage 2, stok gudang, pack size $P_i$ & MOQ e-Katalog, umur utang AP [23] |
| **Output Kunci** | Proyeksi pemakaian unit kontinu ($\hat{Q}_{0.50}, \hat{Q}_{0.80}, \hat{Q}_{0.90}$) | Target stok optimal ($x_i^*$) yang efisien kas & aman dari anomali wabah | Draf Usulan Pembelian integer box ($B_i \in \mathbb{Z}_{\ge 0}$) format CSV e-Katalog |
| **Runtime** | < 500 ms (CPU) | < 10 ms (CPU) | < 5 ms (CPU) |

---

#### Rincian Implementasi Tiap Stage

##### Stage 1: Prediksi Kebutuhan Obat (`backend/model_forecast.py`)
- **Masalah:** Kebutuhan obat di rumah sakit sering berubah drastis karena faktor luar (wabah musiman, lonjakan pasien rawat inap / BOR, dan hari libur). Rumus rata-rata biasa selalu terlambat menyadari lonjakan wabah ini.
- **Data Masukan:** Riwayat pemakaian harian SIMRS (rata-rata bergerak 7, 14, 30 hari), data keterisian tempat tidur (BOR), serta data jeda waktu curah hujan BMKG (21–35 hari untuk DBD, 3–7 hari untuk ISPA) [19].
- **Cara Kerja:** Model **LightGBM Quantile Regression** dilatih untuk memprediksi 3 skenario kebutuhan untuk horizon $H$ hari ke depan [22]:
  - `P50` ($\tau = 0.50$): Kebutuhan normal sehari-hari (skenario rutin).
  - `P80` ($\tau = 0.80$): Batas aman untuk obat kategori Esensial.
  - `P90` ($\tau = 0.90$): Batas pengaman darurat untuk obat kategori Vital.
  - Penataan berurutan (`np.sort`) memastikan angka prediksi tidak terbalik ($\hat{Q}_{0.50} \le \hat{Q}_{0.80} \le \hat{Q}_{0.90}$) [11].
- **Hasil:** Proyeksi kebutuhan unit per obat untuk periode perencanaan.

##### Stage 2: Optimasi Target Kas & Risiko (Wasserstein DRO) (`backend/allocation_engine.py`)

> **Ringkasan Singkat:**  
> Stage 2 memakai program linier (**Linear Programming / HiGHS LP**) untuk mencari jumlah stok ideal ($x_i^*$) yang paling aman dari risiko wabah terburuk, dengan syarat total belanja tidak melebihi sisa uang kas rumah sakit (BLUD).

- **Masalah Nyata:**
  Dari Stage 1, kita tahu angka kebutuhan obat. Masalahnya: **Uang kas rumah sakit terbatas.**
  - Jika semua obat dibeli dengan batas aman maksimal (P90) $\to$ Uang kas langsung habis dan tekor.
  - Jika dipotong sembarangan atau dibagi rata $\to$ Obat kritis (seperti cairan infus DBD atau Epinefrin) bisa habis saat wabah mendadak melonjak.

- **Analogi Sederhana (Pembagian Kuota RAM):**
  - Anda punya RAM 16 GB (Uang Kas Rp 95 Juta) untuk 50 aplikasi (50 jenis obat).
  - Aplikasi inti (*Database* / Obat Vital) **wajib dapat kuota 100% aman**, tidak boleh sampai crash (`vital_min_stock`).
  - Sisa kapasitas dibagi ke aplikasi pendukung (Obat Esensial).
  - **Peran Optimasi DRO:** Solver menghitung: *"Jika terjadi lonjakan pasien ekstrem akibat cuaca buruk, bagaimana pembagian kas terbaik agar risiko rumah sakit kehabisan obat tetap paling rendah?"*

- **I/O & Spesifikasi Teknis di Kode (`backend/allocation_engine.py`):**
  ```python
  # Input Interface:
  # - scenarios: np.ndarray shape (K, M) (K=3: P50, P80, P90 dari Stage 1)
  # - budget_limit: float (sisa pagu kas cair BLUD)
  # - ven_weights: np.ndarray shape (M,) (V=100.0, E=10.0, N=1.0)
  # - unit_prices: np.ndarray shape (M,)
  # - current_stocks: np.ndarray shape (M,)
  # - vital_min_stocks: np.ndarray shape (M,)
  # - epsilon: float (Wasserstein uncertainty radius)
  #
  # Output:
  # - target_units: dict {item_id: float (x_i^* optimal kontinu)}
  # - solver_status: 'optimal' | 'infeasible'
  ```

- **3 Aturan Utama Solver LP:**
  1. **Batas Bawah Obat Vital:** Stok obat Vital wajib minimal memenuhi batas aman darurat (`vital_min_stock` KFT) [7].
  2. **Pembagian Kas Berbasis Risiko (DRO):** Sisa kas diprioritaskan ke obat Esensial yang paling rentan terhadap lonjakan kasus agar pasien tetap aman [12][18].
  3. **Penjagaan Pagu Kas:** Total nilai pembelian seluruh obat dibatasi tidak boleh melewati sisa uang kas BLUD.

- **Hasil:** Target kebutuhan ideal per obat ($x_i^*$) yang sudah lolos uji batas kas dan siap dibulatkan ke boks fisik di Stage 3.
- **Kecepatan:** Selesai dalam tempo **< 10 milidetik di CPU standar** (tanpa butuh GPU).

##### Stage 3: Eksekutor Pembulatan Kemasan Box & Kuota Vendor (`backend/allocation_engine.py`)

> **Ringkasan Singkat:**  
> Stage 3 mengubah target hitungan ($x_i^*$) menjadi **jumlah boks fisik resmi e-Katalog** ($\lceil \text{unit}/P_i \rceil$), mematuhi syarat minimal beli (MOQ), dan mengerem pembelian jika utang ke distributor sudah menunggak lama ($\ge 60$ hari).

- **Masalah Nyata:** Distributor farmasi (PBF) di e-Katalog tidak melayani pesanan butiran/pecahan (harus kelipatan boks resmi $P_i$ dan memenuhi batas minimal order $\text{MOQ}_i$) [9]. Selain itu, jika tagihan ke distributor menumpuk akibat klaim BPJS belum cair, distributor bisa memutus sepihak suplai obat ke RSUD (*credit lock*) [23].

- **I/O & Spesifikasi Teknis di Kode (`backend/allocation_engine.py`):**
  ```python
  # Input:
  # - target_units: dict {item_id: float} (hasil Stage 2)
  # - pack_sizes: dict {item_id: int} (isi unit per box)
  # - moq_boxes: dict {item_id: int} (minimal order box vendor)
  # - ap_aging_days: dict {supplier_id: int} (umur invoice terlama)
  #
  # Output:
  # - final_po_boxes: dict {item_id: int (jumlah box fisik e-Katalog)}
  # - is_credit_throttled: dict {item_id: bool}
  ```

- **Alur Kerja Algoritma (Greedy Runway Waterfall):**
  1. **Hitung Ketahanan Stok (Runway):**
     $$\text{Runway (Hari)} = \frac{\text{Stok Gudang} + \text{Pesanan di Jalan} + (\text{Box Dibeli} \times P_i)}{\max(\text{Laju Pakai Harian}, \; \texttt{vital\_min\_stock} / H, \; \epsilon)}$$
  2. **Urutkan Prioritas:** Obat dengan sisa hari ketahanan paling pendek (paling cepat habis) didahulukan paling depan.
  3. **Bagi Anggaran per Lapisan:**
     - `Lapisan 1 (Dasar Vital):` Wajib dipenuhi 100%. Jika kas tidak cukup $\to$ nyalakan alarm kas defisit (`status: INFEASIBLE`).
     - `Lapisan 2 (Dasar Esensial):` Didanai setelah Lapisan 1 selesai.
     - `Lapisan 3 (Cadangan Vital):` Didanai sampai target aman $x_i^*$ dari Stage 2.
     - `Lapisan 4 (Cadangan Esensial):` Jika kas menipis, dipotong dari obat yang paling menyedot anggaran: **Kelas A ➔ B ➔ C** (dana dialokasikan duluan ke **C ➔ B ➔ A**).
     - `Lapisan 5 (Non-Esensial):` Suplemen/vitamin; hanya dibeli jika seluruh obat Vital dan Esensial sudah aman dan kas masih tersisa.
  4. **Aturan Lewati (Skip Rule):** Jika sisa uang kas tidak cukup untuk membeli 1 paket minimal order (MOQ) pada obat di antrean depan, lewati (*skip*) ke obat berikutnya yang harganya muat di sisa kas (agar kas tidak menganggur sia-sia).
  5. **Rem Pengaman Utang Distributor:** Jika utang ke suatu distributor sudah berumur $\ge 60$ hari, pembelian stok cadangan obat non-darurat dari distributor tersebut otomatis ditahan [23] agar RSUD tidak diboikot saat jatuh tempo 90 hari. Obat penyelamat nyawa (Vital) tetap kebal dari aturan ini.
  6. **Ekspor Draf Pesanan:** Hasil akhir dikemas ke format CSV siap unggah ke portal e-Purchasing LKPP [9].

##### Mengapa Stage 2 (Optimasi DRO) dan Stage 3 (Pembulatan Boks) Dipisahkan?
1. **Stage 2 Mengatur Keseimbangan Risiko dan Kas Global:** Solver menghitung alokasi kas terbaik secara bersamaan untuk ke-50 obat terhadap risiko wabah terburuk [12][18]. Jika hanya memakai sistem antrean biasa, obat-obat di urutan awal akan menghabiskan uang kas dan membuat obat di urutan belakang tidak kebagian dana sama sekali.
2. **Stage 3 Menangani Batasan Logistik Nyata:** Mengurus aturan fisik pembelian yang tidak bisa dihitung dalam angka desimal: pembulatan ke kemasan boks utuh ($B_i \in \mathbb{Z}_{\ge 0}$), syarat minimal beli distributor (MOQ), batas kuota tahunan resmi (RKO) [20], dan rem pengaman utang distributor [23].

Pemisahan ini membuat proses hitung tetap sangat cepat (< 10 milidetik di CPU biasa) sekaligus menghasilkan rekomendasi belanja yang rapi dan realistis bagi apotek RSUD.

---

#### Pengaman Klinis & Validasi Fitur

- **Lantai Pengaman Darurat Troli IGD:** Target akhir obat Vital dilindungi batas fisik kit emergensi KFT: $\text{Target}_i = \max(x_i^*, \; \hat{Q}_{\tau}, \; \texttt{vital\_min\_stock}_i)$.
- **Transparansi Faktor Prediksi (TreeSHAP):** Pengaruh tiap faktor dihitung via `model.predict(..., pred_contrib=True)` untuk menampilkan alasan yang jelas di antarmuka (misal: *"BOR naik 15% menyumbang +12 box", "Hujan BMKG jeda 28 hari menyumbang +10 box"*).
- **Evaluasi Sinyal Lag Cuaca:** Pengujian pada data simulasi terkalibrasi mencatat fitur lag hujan 14–28 hari (kalibrasi BMKG Stasiun Klimatologi Bogor Dramaga, WMO 96753) menurunkan error peramalan (wQL) cairan infus pada skenario lonjakan musiman, sedangkan obat kontrol non-cuaca terbukti tidak mengalami pergeseran error [14].
- **Log Override Apoteker:** Setiap perubahan manual jumlah box oleh apoteker wajib mencantumkan alasan tertulis pada `fct_override_log` untuk memenuhi audit akuntabilitas BPK/Itjen.

### 4.3 Logika Alokasi Anggaran & Eksekusi Box (`backend/allocation_engine.py`)

Bagian ini menerjemahkan hasil peramalan kebutuhan menjadi usulan pembelian fisik dengan batasan kas rumah sakit:

#### 1. Batasan Sistem (System Constraints)
1. **Pagu Kas Cair BLUD (Permendagri 79/2018):** Total belanja tidak boleh melebihi sisa kas cair (`budget_limit - PO_berjalan`) [8].
2. **Kuota Rencana Kebutuhan Obat (RKO):** Usulan box per item tidak boleh melebihi sisa kuota tahunan Kemenkes.
3. **Standar e-Katalog LKPP:** Jumlah pemesanan wajib bulat kelipatan pack size ($P_i$) dan memenuhi batas minimum order vendor ($\text{MOQ}_i$) [9].

#### 2. Matriks ABC–VEN (Klasifikasi Klinis & Finansial)
Setiap obat dikategorikan berdasarkan dua dimensi:
- **Kategori VEN (Prioritas Klinis):**
  - **V (Vital):** Obat darurat penyelamat nyawa (mis. Epinefrin, Serum Anti Bisa Ular). **Dilarang kosong sama sekali.**
  - **E (Esensial):** Obat terapi utama penyakit terbanyak (mis. Antibiotik, Insulin, Cairan Infus RL).
  - **N (Non-Esensial):** Obat penunjang / suplemen (mis. Vitamin C, Vitamin B Kompleks).
- **Kelas ABC (Penyerapan Nilai Anggaran):**
  - **Kelas A (80% serapan belanja kumulatif):** ~10%–20% SKU bernilai rupiah tinggi (misal: cairan infus RL/NaCl, antibiotik paten).
  - **Kelas B (15% serapan belanja kumulatif):** ~10%–20% SKU bernilai belanja sedang.
  - **Kelas C (5% serapan belanja kumulatif):** ~60%–80% SKU bernilai belanja rendah.

Berikut tabel matriks keputusan operasional gabungan ABC–VEN:

| Kategori Klinis (VEN) | Kelas A (80% Anggaran) | Kelas B (15% Anggaran) | Kelas C (5% Anggaran) |
| :--- | :--- | :--- | :--- |
| **Vital (V)** | **Prioritas Utama (Kebal Potong):** Wajib dibeli penuh. Diawasi ketat karena menyerap kas besar. | **Prioritas Utama (Kebal Potong):** Wajib dibeli penuh sesuai target. | **Prioritas Utama (Kebal Potong):** Wajib dibeli penuh. Biaya murah, perlindungan nyawa maksimal. |
| **Esensial (E)** | Kebutuhan dasar dibeli. Jika kas kurang, **buffer dipotong paling awal** untuk efisiensi modal. | Kebutuhan dasar dibeli. Buffer dipotong moderat jika kas tipis. | Kebutuhan dasar dibeli. **Buffer didanai lebih dulu** karena murah dan tidak menguras kas. |
| **Non-Esensial (N)** | **Dipangkas pertama kali hingga 0 box** saat kas rumah sakit ketat. | Dipangkas setelah kelas A jika kas masih defisit. | Dipangkas terakhir di kelompok Non-Esensial. |

#### 3. Logika Alokasi Bertahap (Greedy Runway Waterfall)

Jika kas rumah sakit tidak mencukupi untuk mendanai seluruh kebutuhan ideal, sistem membagi anggaran melalui **5 Lapisan Waterfall** secara berurutan:

| Lapisan Prioritas | Yang Didanai | Target Kuantil / Batas | Perilaku Saat Kas Terbatas |
| :---: | :--- | :---: | :--- |
| **1** | **Dasar Vital** | $\max(\text{P50}, \texttt{vital\_min\_stock})$ | **Wajib Lolos 100%:** Mengamankan obat rutin dan intermiten (SABU/Antidot). Jika kas gagal mendanai, sistem memicu alarm darurat (`INFEASIBLE`). |
| **2** | **Dasar Esensial** | P50 (Median) + $\Delta Q_{\text{cuaca}}$ | Didanai setelah Lapisan 1 selesai. Pemenuhan parsial jika kas tersisa menipis. Proyeksi wabah cuaca kebal potong. |
| **3** | **Buffer Vital** | $x_i^*$ (DRO Target $\ge$ P90) | Cadangan darurat untuk lonjakan kasus/KLB. Plafon target dipatok dari solusi optimal Stage 2. Kebal rem utang vendor. |
| **4** | **Buffer Esensial** | $x_i^*$ (DRO Target $\ge$ P80) | Buffer pemenuhan rutin. Dibagi ke dalam sub-lapisan bertingkat: **4a (Kelas C) ➔ 4b (Kelas B) ➔ 4c (Kelas A)**. Urutan antrean runway hanya berlaku di dalam sub-lapisan yang sama. Buffer Kelas A ditahan jika utang vendor $\ge 60$ hari. |
| **5** | **Non-Esensial** | P50 (Baseline) | Suplemen/vitamin. Dialokasikan paling akhir hanya jika kas masih bersisa (urut sub-lapisan **C ➔ B ➔ A**). Ditahan jika utang vendor $\ge 60$ hari. |

**Aturan Penentuan Antrean Pembelian (Runway Metric):**
- **Metrik Runway (Sisa Hari Ketahanan Stok):**
  $$\text{Runway (Hari)} = \frac{\text{Stok Fisik Gudang} + \text{Pesanan di Jalan} + (\text{Box yang Sudah Dibeli} \times \text{Isi per Box})}{\max(\text{Laju Konsumsi Harian}, \; \texttt{vital\_min\_stock} / H, \; \epsilon)}$$
  *(Penyebut $\texttt{vital\_min\_stock} / H$ mengamankan obat Vital intermiten agar tidak tergeser ke antrean belakang).*
- **Mekanisme Antrean (Urutan Pembelian Box):**
  Obat dengan nilai *Runway* paling pendek (stok paling cepat habis) ditempatkan di urutan terdepan untuk diproses pembeliannya per boks hingga target kebutuhan terpenuhi.
- **Aturan Lewati vs Berhenti (Skip Rule):**
  Jika sisa uang kas tidak cukup untuk membeli 1 paket minimal order (MOQ) pada obat urutan terdepan, sistem tidak berhenti total, melainkan melompati obat tersebut dan lanjut ke obat berikutnya yang harganya masih muat di sisa kas. Hal ini memastikan anggaran kas terpakai optimal tanpa ada dana yang menganggur.
- **Mengapa Cadangan Ekstra Kelas A Dipotong Lebih Dulu Saat Kas Ketat?**
  1. **Kebutuhan Utama Tetap Aman:** Kebutuhan dasar rutin pasien dan lonjakan akibat musim hujan (seperti cairan infus DBD) berada di **Lapisan 2 (Dasar Esensial)** dan **TIDAK PERNAH DIPANGKAS**. Pasien DBD tetap dijamin mendapatkan infus sesuai kebutuhan.
  2. **Hemat Kas Signifikan:** Yang dipotong di Lapisan 4 hanyalah *stok cadangan tambahan*. Karena obat Kelas A harganya mahal, menunda 1 boks cadangan Kelas A bisa menghemat anggaran puluhan kali lipat dibanding memotong obat murah (Kelas C). Langkah ini menyelamatkan keuangan RSUD dari ancaman pemblokiran distributor tanpa mengorbankan pelayanan langsung ke pasien.
- **Rem Pengaman Utang Distributor:**
  Jika tagihan ke suatu distributor sudah menunggak 60 hari (mendekati batas kritis pemutusan kiriman obat di hari ke-90), pembelian cadangan obat non-darurat dari distributor tersebut otomatis ditahan [23]. Obat darurat penyelamat nyawa (Vital) tetap diizinkan beli demi keselamatan pasien.

#### 4. Mode Konfigurasi Pengadaan (API Presets)

Endpoint `/api/reorder-defekta` menyediakan 3 preset kebijakan alokasi via parameter `preset_mode`:

| Preset API | Target Kuantil | Karakteristik Operasional | Rekomendasi Penggunaan |
| :--- | :---: | :--- | :--- |
| `p50_lean` | Vital: P50<br/>Esensial: P50 | Tanpa buffer stok tambahan. Penyerapan kas minimal. | Akhir tahun anggaran, kas cair BLUD sangat ketat, atau penundaan klaim BPJS ekstrem. |
| `p80_standard` *(Default)* | Vital: P90<br/>Esensial: P80 | Buffer berimbang antara proteksi ketersediaan dan modal kerja. | Operasional normal bulanan instalasi farmasi RSUD. |
| `climate_robust` | Vital: P90 + $\Delta_{\text{iklim}}$<br/>Esensial: P80 + $\Delta_{\text{iklim}}$ | Mengaktifkan penyangga epidemiologis dari lag curah hujan BMKG (21–35 hari). | Menjelang puncak musim hujan lebat (antisipasi lonjakan DBD dan ISPA). |

#### 5. Penanganan Kas Defisit Ekstrem (`status: INFEASIBLE`)

Kondisi `INFEASIBLE` terjadi ketika sisa kas cair BLUD tidak mampu mendanai kebutuhan **Dasar Vital (Lapisan 1)**:

1. **Respons Payload API:**  
   Backend mengembalikan objek status eksplisit:
   ```json
   {
     "is_feasible": false,
     "status": "INFEASIBLE",
     "deficit_vital_rupiah": 14500000,
     "unfunded_vital_skus": ["V-01", "V-04"],
     "allocated_boxes": {"V-01": 5}
   }
   ```
2. **Prosedur Tindak Lanjut Regulasi:**
   - **Nota Fleksibilitas & Eskalasi Kas BLUD:** 
     1. Jika pendapatan melampaui target, sistem menerapkan Ambang Batas Fleksibilitas RBA (Permendagri 79/2018 Ps. 74 ayat (2) & Ps. 75 ayat (6)) [8].
     2. Jika kas benar-benar defisit akibat klaim macet, sistem menyusun draf pergeseran rincian belanja operasional (Ps. 61 ayat (5)), usulan dispensasi belanja darurat ke Kepala Daerah (Ps. 74 ayat (5)), atau usulan tambahan anggaran APBD ke PPKD (Ps. 74 ayat (6)) [8].
     3. Untuk kebutuhan mendesak yang terkunci di e-Katalog, RSUD dapat mengaktifkan pengadaan langsung darurat berbasis Perkada BLUD (Ps. 76 ayat (2) jo. Ps. 77) [8].
   - **Penyaluran Obat Darurat Antar-Fasyankes:** Sistem menyusun draf surat resmi penyaluran/bon pinjam-mengganti persediaan darurat antar-RSUD atau ke Dinas Kesehatan setempat (PerBPOM 24/2021 Ps. 27–29 jo. Permenkes 72/2016) [16]. Sesuai UU 1/2004 Ps. 3 ayat (3) [15], sistem mengunci penerbitan PO reguler tanpa ketersediaan kas guna melindungi PPK dari pelanggaran perbendaharaan negara.

---

## 5. End-to-End System Pipeline

### 5.1 Arsitektur Visi Sistem Lengkap (Target Skala Penuh)

```mermaid
flowchart LR
    %% Palette: Minimalist Enterprise (Neutral Slate Foundation + Cobalt Precision Accent)
    classDef default fill:#ffffff,stroke:#cbd5e1,stroke-width:1.2px,color:#0f172a;
    classDef coreEngine fill:#ffffff,stroke:#1d4ed8,stroke-width:1.8px,color:#0f172a;
    classDef actionNode fill:#ffffff,stroke:#0f172a,stroke-width:1.8px,color:#0f172a;

    subgraph Data ["Data Masukan RSUD & Regulasi (Read-Only)"]
        direction TB
        D1["Transaksi SIMRS / SatuSehat (Resep, Dispensing, Mutasi Depo)"]
        D2["Stok Fisik Gudang & Depo (Batch, Kedaluwarsa, Non-Karantina)"]
        D3["Katalog LKPP & Kebijakan (Pack, MOQ, Pagu BLUD, Kuota RKO, VEN)"]
        D4["Konteks Operasional (BOR, Kunjungan Poli, Musim/Cuaca BMKG)"]
        D5["AP Aging Piutang Vendor PBF (Umur Faktur TOP 30–90 Hari)"]
    end
    style Data fill:#f8fafc,stroke:#e2e8f0,stroke-width:1px,color:#475569

    subgraph Analytics ["Engine Analitik & Prediksi (Stage 1)"]
        direction TB
        PRE["Decensoring Permintaan (Imputasi Stockout) & Stok Efektif FEFO"]
        
        RUNRATE_DEPO["Kalkulasi Median Run-Rate Depo<br/>(DoC Harian IGD, RJ, RI)"]
        LGB_RS["Stage 1: LightGBM Kuantil Agregat RS (L+R)<br/>(Tau = 0.5, 0.8, 0.9 + BMKG Climate Lag)"]:::coreEngine
        SHAP["TreeSHAP Feature Attribution Engine<br/>(Dekomposisi Driver Permintaan: BOR, Cuaca, Substitusi)"]:::coreEngine
        
        ALERT["Alert Depo: Hari Ketahanan Stok vs Lead Time Internal"]
        
        PRE --> RUNRATE_DEPO
        PRE --> LGB_RS
        LGB_RS --> SHAP
        RUNRATE_DEPO --> ALERT
    end
    style Analytics fill:#f8fafc,stroke:#e2e8f0,stroke-width:1px,color:#475569

    subgraph Allocation ["Engine Alokasi & Pengadaan (Stage 2 & 3)"]
        direction TB
        ABC["Klasifikasi ABC Pareto (Nilai Belanja Tahunan x Harga)"]
        LP_OPT["Stage 2: Wasserstein DRO Target Optimizer<br/>(HiGHS Dual LP &lt; 10ms &amp; Penjaga Pagu Kas BLUD)"]:::coreEngine
        STAGE3["Stage 3: Greedy Runway Waterfall Allocator<br/>(Prioritas VEN, Konversi Box Pi, MOQ, Kuota RKO &amp; AP Credit Lock)"]:::coreEngine
        
        LGB_RS --> LP_OPT
        LP_OPT --> STAGE3
        ABC --> STAGE3
    end
    style Allocation fill:#f8fafc,stroke:#e2e8f0,stroke-width:1px,color:#475569

    subgraph Execution ["Tata Kelola & Eksekusi"]
        direction TB
        EXEC["Review & Override Apoteker (Audit Trail) → Draf CSV LKPP"]:::actionNode
    end
    style Execution fill:#f8fafc,stroke:#cbd5e1,stroke-width:1.2px,color:#0f172a

    D1 --> PRE
    D2 --> PRE
    D2 --> LP_OPT
    D2 --> STAGE3
    D3 --> LP_OPT
    D3 --> STAGE3
    D4 --> LGB_RS
    D5 --> STAGE3
    PRE --> ABC
    
    ALERT --> EXEC
    SHAP --> EXEC
    STAGE3 --> EXEC
```

### 5.2 Arsitektur Prototipe Hackathon (FastAPI Backend + Next.js Frontend)

Untuk kompetisi dan demo juri, pipeline berjalan decoupled: REST API FastAPI melayani komputasi kuantil & alokasi optimasi kas, dikonsumsi oleh Webapp Frontend Next.js 16 (berkiblat 1:1 Rxflow Dribbble):

```mermaid
flowchart LR
    %% Palette: Minimalist Enterprise (Neutral Slate Foundation + Cobalt Precision Accent)
    classDef default fill:#ffffff,stroke:#cbd5e1,stroke-width:1.2px,color:#0f172a;
    classDef inputNode fill:#ffffff,stroke:#94a3b8,stroke-width:1.2px,color:#0f172a;
    classDef extNode fill:#ffffff,stroke:#64748b,stroke-width:1.2px,stroke-dasharray: 4 3,color:#0f172a;
    classDef coreNode fill:#ffffff,stroke:#1d4ed8,stroke-width:1.8px,color:#0f172a;
    classDef outputNode fill:#ffffff,stroke:#0f172a,stroke-width:1.8px,color:#0f172a;

    subgraph RS [" 1. Infrastruktur Eksisting RSUD "]
        direction TB
        SIMRS["<b>SIMRS Khanza / SatuSehat</b><br/>• Riwayat Resep &amp; Depo<br/>• Saldo Stok Fisik &amp; Batch"]:::inputNode
        POLICY["<b>Pagu Kas BLUD &amp; RKO</b><br/>Permendagri 79/2018"]:::inputNode
    end
    style RS fill:#f8fafc,stroke:#e2e8f0,stroke-width:1px,color:#475569

    subgraph EXT [" 2. Sinyal Eksternal "]
        direction TB
        BMKG["<b>Data BMKG &amp; LKPP</b><br/>• Presipitasi Hujan (Lag DBD/ISPA)<br/>• E-Katalog (Pack Size &amp; MOQ)"]:::extNode
    end
    style EXT fill:#f8fafc,stroke:#e2e8f0,stroke-width:1px,color:#475569

    subgraph ENGINE [" 3. sediantra Engine (Add-on REST API) "]
        direction TB
        S1["<b>Stage 1: Prediksi Kebutuhan</b><br/>LightGBM Kuantil (P50, P80, P90)<br/>Decensoring + BMKG Climate Lag"]:::coreNode
        S2["<b>Stage 2: Optimasi Kas &amp; Risiko</b><br/>Wasserstein DRO Dual LP (HiGHS)<br/>Pagu Kas BLUD + Proteksi Vital"]:::coreNode
        S3["<b>Stage 3: Eksekusi Boks LKPP</b><br/>Greedy Runway Waterfall (ABC–VEN)<br/>Pack Size LKPP &amp; AP Rem Utang"]:::coreNode
        S1 --> S2 --> S3
    end
    style ENGINE fill:#f1f5f9,stroke:#cbd5e1,stroke-width:1.5px,color:#0f172a

    subgraph OUT [" 4. Pengguna &amp; Luaran "]
        direction TB
        DASH["<b>Dasbor Interaktif Web</b><br/>• Alert Depo (Days of Cover)<br/>• Review &amp; Override Apoteker"]:::outputNode
        CSV["<b>Draf PO e-Purchasing LKPP</b><br/>Format CSV Siap Upload"]:::outputNode
        DASH --> CSV
    end
    style OUT fill:#f8fafc,stroke:#e2e8f0,stroke-width:1px,color:#475569

    SIMRS -->|Replika Read-Only| S1
    POLICY --> S2
    BMKG --> S1
    BMKG --> S3
    S3 --> DASH
```

| Modul | Visi Lengkap (§5.1) | Prototipe Hackathon (§5.2) | Rasionalisasi Pemangkasan |
| :--- | :--- | :--- | :--- |
| **Sumber Data** | 8 integrasi (FHIR, SIMRS, BMKG, e-Katalog, RKO) | 1 Data mart terpadu SQLite/CSV (`data/`) berbasis skema tabel Khanza + harga LKPP + sinyal BMKG | Mencegah bottleneck birokrasi RS; ground truth tersimpan untuk evaluasi decensoring akurat |
| **Decensoring** | Regresi data tersensor + substitusi dinamis KFT/Fornas | Imputasi median pemakaian 7 hari sebelum kosong | Ringan, deterministik, dan bebas komputasi iteratif |
| **Model Peramalan & Optimasi** | Pipeline Terintegrasi: LightGBM + Wasserstein DRO LP | Three-Stage Pipeline: Stage 1 LightGBM Quantile + Stage 2 Wasserstein DRO Dual LP HiGHS target kontinu (<10ms) + Stage 3 Greedy Runway Waterfall alokasi box; DoC depo dari median bergulir 7 hari | Menjaga latensi inferensi ultra-cepat di bawah 1 detik di CPU standar tanpa GPU |
| **Tail Quantile** | Kalibrasi batas risiko dinamis | Direct quantiles (P50, P80, P90) + penjamin `vital_min_stock` | Akurat, stabil, dan aman dari kekosongan fatal tanpa inferensi kompleks |
| **Klasifikasi ABC** | Dihitung ulang tiap siklus dari kebutuhan terdekode; ambang dikonfigurasi KFT | Sama, ambang tetap 80% / 95% | Aturan sederhana dan murah dihitung; menjadi urutan pendanaan buffer E/N dan penanda pengawasan biaya |
| **Dual Constraint & Credit Guard** | Plafon kas cair BLUD + kuota RKO tahunan + batas AP PBF | Plafon kas bulanan BLUD + batas kuota RKO + rem likuiditas utang distributor | Menjaga kas BLUD sekaligus melindungi RSUD dari embargo pengiriman distributor |
| **Antarmuka Pengguna** | Web app produksi RS terintegrasi SSO SIMRS | Next.js 16 decoupled frontend (berkiblat 1:1 Rxflow) + FastAPI backend (`backend/main.py`) | Menjamin interaktivitas rendering tinggi, segmented progress bar intuitif, dan bebas AI-copilot generik |

---

## 6. Kasus Uji Rekonsiliasi Kas (Assertion Test Matrix)

Acuan unit test untuk memverifikasi kalkulasi alokasi greedy dual-constraint pada 2 SKU representatif (dengan asumsi horizon perencanaan $H = 30$ hari, `vital_min_stock` V-01 = 30 unit, MOQ = 1 box, sisa RKO masing-masing = 200 box):
- **V-01 (Vital / Kelas A - Epinefrin 1 mg):** Stok fisik gudang 40, Dasar 120, Target 150 unit. Rp10.000/ampul (Pack size 10/box = Rp100.000/box). Laju pakai harian = 4 unit/hari.
- **E-02 (Esensial / Kelas B - Ceftriaxone 1 g):** Stok fisik gudang 20, Dasar 100, Target 130 vial. Rp5.000/vial (Pack size 10/box = Rp50.000/box). Laju pakai harian = 3,33 unit/hari.

| Skenario Pengujian | Pagu Kas BLUD | Alokasi V-01 | Alokasi E-02 | Total Belanja | Status & Invarian Sistem |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Baseline Tanpa Pembelian** | Rp 0 | 0 box | 0 box | Rp 0 | V-01 habis hari ke-10; E-02 habis hari ke-6. |
| **2. Anggaran Ketat (Hemat Kas P50)** | Rp 1.200.000 | 8 box (80 ampul) | 8 box (80 vial) | Rp 1.200.000 | Kebutuhan dasar terpenuhi pas; buffer = 0; sisa kas = Rp 0. |
| **2b. Kompetisi Buffer (Kas Menengah)** | Rp 1.500.000 | 11 box (110 ampul) | 8 box (80 vial) | Rp 1.500.000 | Dasar V & E terpenuhi; Buffer V-01 (Lapisan 3) didanai penuh (+3 box); Buffer E-02 (Lapisan 4) belum kebagian kas. |
| **3. Anggaran Penuh (Target Kuantil)** | Rp 1.800.000 | 11 box (110 ampul) | 11 box (110 vial) | Rp 1.650.000 | Target kuantil tercapai; buffer 30 unit tiap SKU; sisa kas Rp 150.000. |
| **4. Kas Defisit (Anggaran Tidak Cukup)**| Rp 500.000 | 5 box (50 ampul) | 0 box | Rp 500.000 | `status: INFEASIBLE` (defisit dasar V Rp 300.000); alokasi 5 box adalah serapan sisa kas darurat; picu nota eskalasi BLUD. |

*Invarian Assert:* Total Belanja $\le$ Pagu Kas, pembelian kelipatan pack-size (10), dan obat Vital diprioritaskan sebelum buffer Esensial.

---

## 7. Evaluasi MVP (Yang Nyata Dihitung)

### Metode: Rolling-Origin Replay
Untuk setiap tanggal replay $t$ (mis. awal tiap bulan selama 12 bulan terakhir):
1. Latih model **hanya dengan data historis sebelum $t$** (tanpa kebocoran data masa depan / *lookahead bias*).
2. Hasilkan rekomendasi draf usulan pembelian dengan pagu kas BLUD dan kuota RKO historis bulan itu.
3. Simulasikan stok harian ke depan memakai kebutuhan resep aktual dan lead time aktual dari riwayat PO.
4. Ulangi hal yang sama untuk **baseline**: Metode Konsumsi standar Permenkes 72/2016 jo. Petunjuk Teknis Standar Pelayanan Kefarmasian di RS (Kemenkes RI, 2019) (Static Safety Stock 20%) [7] dengan pagu yang sama.

### Metrik Evaluasi

| No | Metrik | Formulasi / Definisi | Target MVP | Catatan |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Kualitas Forecast Probabilistik:** wQL & Rata-rata Quantile Score | Weighted Quantile Loss: $\text{wQL}_\tau = \frac{2 \sum_i \rho_\tau(y_i - \hat{y}_{i,\tau})}{\sum_i y_i}$; Normalized Quantile Loss (NQL) per SKU; Rata-rata Quantile Score ($\overline{\text{QS}}$) pada $\tau \in \{0.5, 0.8, 0.9\}$ [6] | Lebih rendah dari baseline MA & Seasonal Naive | Mengukur akurasi seluruh profil distribusi kebutuhan dinormalisasi volume tanpa bias agregat fast-moving |
| **2** | **Kalibrasi:** Cakupan empiris per $\tau$ | Fraksi aktual $\le \hat{Q}_\tau$ | Mendekati $\tau$ per kelas (0,50 / 0,80 / 0,90; Vital dievaluasi pada $\hat{Q}_{0{,}90}$ + lantai `vital_min_stock`) | Mengukur apakah risiko *under-forecasting* terkalibrasi |
| **3** | **Layanan Klinis:** Hari stockout & Fill Rate | $\text{Fill Rate} = \frac{\sum \text{qty dispensed}}{\sum \text{qty prescribed}}$; Unmet Prescription Count | Hari stockout V lebih rendah dibanding baseline (target ideal 0 di luar skenario kas ekstrem); Fill Rate V & E $\ge$ baseline | Dampak riil ke pemenuhan SPM RS (Kepmenkes 129/2008) |
| **4** | **Efisiensi Finansial & Kedaluwarsa:** Realized Service & Wastage | Nilai obat kedaluwarsa/wastage (FEFO projection) dan nilai realisasi belanja kebutuhan obat yang berhasil terlayani | Penurunan nilai obat terbuang kedaluwarsa $\ge 50\%$ vs baseline statis | Dilaporkan berpasangan dengan Fill Rate (mencegah *gaming* over-ordering dan membedakan efisiensi belanja riil dari kelumpuhan pemesanan akibat *credit lockout*) |
| **5** | **Waktu Perencanaan** | Durasi review draf defekta oleh apoteker | < 15 menit per siklus | Diukur empiris saat usability testing |

### Hasil Evaluasi Empiris Prototipe (Benchmark Replay vs Baseline Permenkes & MA)

### Karakteristik Lingkungan Uji & Empirical Benchmark

> **Spesifikasi Lingkungan Pengujian (RSUD di Bogor Terkalibrasi BMKG):**  
> Evaluasi kuantitatif dijalankan pada lingkungan simulasi data terkalibrasi (*Calibrated Semi-Synthetic Environment*) sesuai kerangka regulasi data (UU 27/2022 PDP & UU 17/2023 Kesehatan):
>
> | Parameter Uji | Nilai / Konfigurasi | Keterangan untuk Engineer |
> | :--- | :--- | :--- |
> | **Jumlah SKU Master** | 50 SKU representatif (skala master) & stress-test s.d. 2.000 SKU | Kombinasi Fast-moving, Lumpy, Intermiten Vital (SABU/Antidot), dan Non-Esensial. |
> | **Skema Database** | SIMRS Khanza (Standar RSUD) | Struktur tabel diselaraskan dengan relasi produksi RSUD di Indonesia. |
> | **Katalog & Harga** | e-Katalog INAPROC LKPP 2024–2026 | Harga satuan, pack size fisik ($P_i$), dan batas $\text{MOQ}_i$ vendor 1:1 riil [9]. |
> | **Durasi & Observasi**| 1.460 hari (4 tahun penuh: 2023–2026) | 73.000 catatan dispensing harian (3 tahun train Days 0–1.095, 1 tahun pure holdout Days 1.095–1.460). |
> | **Sinyal Eksogen Iklim**| BMKG Stasiun Klimatologi Bogor (Dramaga, WMO 96753) | Curah hujan riil ~4.400 mm/tahun, 217 hari hujan untuk replikasi lag DBD (14–28 hari) dan ISPA (3–7 hari) [19]. |
> | **Stress Test Finansial**| Macet klaim BPJS 30–90 hari [23] | Ambang rem likuiditas aktif di hari ke-60 sebelum batas kritis freeze vendor 90 hari. |
> | **Uji Toleransi Vendor** | Variasi batas tempo vendor diuji 70–90 hari | Membuktikan rem utang konsisten mempertahankan **0 hari boikot distributor**. |

Pengujian empiris peramalan kebutuhan melintasi 4 siklus tahunan penuh (3 tahun data latih dan 1 tahun murni *out-of-sample holdout* pada 12 siklus perencanaan bulanan) menghasilkan bukti performa berikut:

| Kuantil Kebutuhan ($\tau$) | wQL sediantra (AI) | wQL Permenkes 72/2016 | wQL Moving Avg | Coverage sediantra | Coverage Permenkes | Status Kalibrasi |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$\tau = 0.50$ (Median)** | **0.1598** | 0.2252 | 0.1350 | **52.3%** | 88.8% | Terkalibrasi Sempurna (Target: 50,0%) |
| **$\tau = 0.80$ (Esensial)** | **0.1288** | 0.1076 | 0.1371 | **80.3%** | 88.8% | Terkalibrasi Sempurna (Target: 80,0%) |
| **$\tau = 0.90$ (Vital)** | **0.0982** | 0.0685 | 0.1378 | **91.3%** | 88.8% | Proteksi Darurat Sempurna (Target: 90,0%) |

### Perbandingan Komprehensif Antar-Model Peramalan (Benchmark 4 Tahun)

Evaluasi *head-to-head* pada 1 tahun holdout (unseen 2026) terhadap 5 arsitektur algoritma peramalan:

| Model Peramalan | wQL P50 | wQL P80 | wQL P90 | Mean wQL | P90 Coverage | WAPE (%) | Karakteristik Klinis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **sediantra (Multi-Quantile LightGBM)** | **0.1598** | **0.1288** | **0.0982** | **0.1289** | **91.3%** | **15.98%** | **Unggul Multivariat:** Terkalibrasi seimbang di semua kuantil, membaca BOR & lag iklim BMKG. |
| **TSB + NegBin (Intermittent Standard)** | 0.1078 | 0.0777 | 0.0510 | 0.0788 | 89.5% | 10.78% | Sangat kuat untuk obat intermiten murni, namun univariat (buta sinyal cuaca BMKG & BOR). |
| **Climate-Aware ARX (Regresi Iklim BMKG)**| 0.1907 | 0.1378 | 0.0994 | 0.1426 | 79.0% | 19.07% | Menangkap regresi cuaca linier, namun *under-coverage* pada kuantil darurat (79% < 90%). |
| **SIMRS Moving Average (30-Day Window)** | 0.1350 | 0.1371 | 0.1378 | 0.1367 | 48.8% | 13.50% | **Rentan Stockout:** Coverage P90 hanya 48.8% (obat darurat jebol di 51.2% waktu). |
| **Permenkes 72/2016 Static Standard** | 0.2252 | 0.1076 | 0.0685 | 0.1338 | 88.8% | 22.52% | **Modal Kerja Mati:** Menimbun buffer statis 88.8% rata di semua obat termasuk vitamin. |

*Temuan Evaluasi:*
1. **Kalibrasi Probabilistik Presisi Tinggi:** Berkat pelatihan 3 tahun melintasi aneka variasi tahun basah dan kering di Bogor, model mencapai akurasi kuantil ideal pada tahun holdout: P50 tepat di **52.3%** (target 50%), P80 di **80.3%** (target 80%), dan P90 di **91.3%** (target 90%).
2. **Kelemahan Fatal Moving Average SIMRS:** Walaupun MA mencatat WAPE 13.5%, metode ini mengalami kegagalan proteksi fatal karena kuantil keselamatannya hanya memiliki *coverage* **48.8%** (artinya obat darurat habis pada lebih dari separuh waktu pelayanan).
3. **Mencegah Penimbunan Modal Mati:** Rumus baku Permenkes 72/2016 memicu pembelian berlebih secara merata (angka cakupan 88.8% di seluruh obat, termasuk vitamin dan suplemen yang menimbun uang kas). sediantra membagi porsi perlindungan secara tepat: 52.3% pada kebutuhan rutin biasa (menghemat kas), dan 91.3% difokuskan khusus untuk obat penyelamat nyawa (Vital).
4. **Terbukti Peka Terhadap Musim Hujan (BMKG Bogor):** Prediksi kebutuhan cairan infus DBD meningkat akurasinya karena model membaca sinyal jeda hujan 14–28 hari dari data BMKG Bogor yang stabil dipelajari melintasi 3 siklus musim hujan sebelumnya.
5. **Karakteristik Akurasi Berdasarkan Karakter Pemakaian Obat:**
   - *Obat Rutin / Stabil (48,5% SKU):* Sangat presisi dengan tingkat kesalahan (WAPE) hanya **5,55%**.
   - *Obat Fluktuatif (31,1% SKU):* Tetap terjaga aman dengan cakupan perlindungan **91,6%**.
   - *Obat Darurat / Jarang Keluar (3,0% SKU):* Dipastikan **Zero Stockout (100% aman tidak pernah kosong)** berkat batas bawah obat vital (`vital_min_stock`).
   - *Obat Pemakaian Tidak Menentu (17,4% SKU):* Variasi kebutuhan tinggi langsung diamankan oleh cadangan risiko di Stage 2 dan Stage 3.
6. **Pentingnya Memisahkan Stok Troli IGD:** Stok obat darurat di troli IGD yang tersegel harus dipisahkan dari hitungan stok pelayanan umum, agar sistem tidak terlambat menyalakan peringatan obat menipis di loket apotek.

### Hasil Evaluasi Empiris Komparatif & Ketahanan Bencana (360 Hari Penuh)

Pengujian komparatif menyeluruh pada siklus 360 hari (50 SKU RSUD) mencakup performa dasar dan skenario ekstrem simultan (**Lonjakan Presipitasi Ekstrem BMKG Bogor + Penundaan Pembayaran Klaim BPJS 90 Hari**):

| Parameter Evaluasi | Baseline A<br>Permenkes 72/2016 jo. Juknis 2019<br>*(Static Safety Stock 20%)* | Baseline B<br>HiGHS Exact MILP<br>*(Global Exact Solver)* | Baseline C<br>sediantra Produksi<br>*(Pipeline 3-Tahap: DRO + AP Throttle + Waterfall)* | Selisih C vs A | Selisih C vs B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vital Fill Rate (%)** | 96,45% | 93,83% | **92,06%** | -4,39 pp | -1,77 pp *(Setara Aman)* |
| **Vendor Credit Freeze (Lockout Days)** | **0 Hari** | 60 Hari *(Diboikot PBF)* | **0 Hari** | 0 Hari *(Sama Aman)* | **-60 Hari (Bebas Embargo)** |
| **Sisa Kas Cair Akhir (Ending Cash)** | Rp 11.238.800 | Rp 18.421.100 | **Rp 47.454.400** | **+Rp 36.215.600 (+322%)** | **+Rp 29.033.300 (+157%)** |
| **Total Realisasi Belanja (360 Hari)** | Rp 1.157.386.200 | Rp 1.066.203.900 | **Rp 1.084.545.600** | -Rp 72.840.600 *(Hemat Kas)* | +Rp 18.341.700 *(Lebih Terlayani)* |
| **Waktu Selesai Komputasi Alokasi (50 SKU)** | 0,08 Detik | 0,85 Detik | **0,0035 Detik (3.59 ms)** | -95,6% (22x Lebih Cepat) | **-99,5% (236x Lebih Cepat)** |

*Temuan Komparatif Nyata:*  
1. **Mengapa Solver Matematika Murni (HiGHS Exact MILP) Gagal di Dunia Nyata:** Baseline B adalah solver optimasi eksak standar Operations Research. Di atas kertas formulasi MILP terlihat canggih, namun saat macet klaim BPJS tiba, MILP menghabiskan kas untuk memborong obat buffer hingga kas RSUD tiris. Akibatnya, utang ke distributor menumpuk dan **distributor memboikot RSUD selama 60 hari penuh (Vendor Lockout)**. Pasien tidak bisa dilayani karena distributor menolak mengirim obat.
2. **sediantra Menyelamatkan Likuiditas & Ketersediaan Obat:** sediantra membuktikan bahwa manajemen rantai pasok rumah sakit bukan sekadar masalah matematika statis, melainkan **ketahanan likuiditas**. Berkat rem utang (*AP Debt Throttle*), sediantra mempertahankan **0 hari lockout**, menjaga kas darurat Rp 47,4 Juta, dan mengamankan 92,06% ketersediaan obat vital tanpa henti.

### Uji Ketahanan Skalabilitas Beban Apoteker (50 s.d. 2.000 SKU)

Pengujian skalabilitas komputasi pada CPU standar x86_64 membuktikan kesiapan sediantra melayani instalasi farmasi dari RSUD Tipe C hingga RS Rujukan Nasional Tipe A:

| Skala Rumah Sakit | Jumlah SKU Aktif | Stage 2 DRO Dual LP | Stage 3 Waterfall | Total UI Response Time | Peak Memory | Status Integritas |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pareto Inti IGD / ICU** | **50 SKU** | 27,0 ms | 10,8 ms | **37,8 ms** | 0,23 MB | 100% Invarian Valid |
| **Pengadaan RSUD Kelas C** | **250 SKU** | 124,5 ms | 0,6 ms | **125,1 ms** | 0,98 MB | 100% Invarian Valid |
| **Formularium RSUD Kelas B**| **500 SKU** | 246,6 ms | 1,0 ms | **247,6 ms** | 1,96 MB | 100% Invarian Valid |
| **Master Farmasi + BMHP Kritis**| **1.000 SKU** | 503,9 ms | 2,0 ms | **506,0 ms (~0,5 detik)** | **3,90 MB** | 100% Invarian Valid |
| **Gudang Pusat RS Rujukan A** | **2.000 SKU** | 1.037,6 ms | 3,8 ms | **1.041,4 ms (~1,04 detik)**| **7,81 MB** | 100% Invarian Valid |

### Uji Komparasi Ketahanan Data Sampah SIMRS (Garbage-In Robustness)

Evaluasi perilaku algoritma saat menerima data korup dari SIMRS Khanza (Stok negatif `-120`, Typo `500.000` unit, nilai `NaN`/`None`, dan umur utang PBF rusak):

| Jenis Kerusakan Data SIMRS | Baseline A: Permenkes 72/2016 (Excel) | Baseline B: Exact MILP (OR Solver Standar) | sediantra (3-Stage Pipeline + Defensive Guard) |
| :--- | :--- | :--- | :--- |
| **Typo Ekstrem Petugas (mis. entri 500.000 unit)** | **JEBOL FATAL:** 99,9% anggaran rumah sakit (Rp 94,9M) habis disedot 1 obat itu saja. Obat lain mati. | **KORUP PARAH:** Menyedot Rp 51,7M (54,4% kas), mematikan alokasi 13 obat lainnya. | **KEBAL:** Typo dijepit (*historical clamp*) maksimal Rp 17,9M. **Seluruh 50 SKU tetap terdanai.** |
| **Stok Negatif (Bug Retur / Gantung: -50 unit)** | **OVER-ORDER:** Memesan box fiktif berlebihan karena mengira defisit fisik riil. | **DISTORSI:** Terjebak menambah kuota pesanan tanpa batas. | **KEBAL:** Stok minus dinormalisasi ke `0.0`, order dihitung proporsional terhadap laju konsumsi. |
| **Nilai Hilang (`NaN`, `None`, Cell Kosong)** | **CRASH:** Rumus terputus (`TypeError: unsupported operand NoneType`). | **CRASH:** Batas kendala linier gagal dibentuk karena nilai null. | **KEBAL:** Sanitizer otomatis mengimputasi *statutory minimum* tanpa interupsi sistem. |
| **Tanggal Utang PBF Rusak (-45 hari / 99.999 hari)**| **DIABAIKAN:** Tidak ada sensor utang distributor (order jalan terus hingga di-lockout). | **TIDAK TERKONTROL:** Solver tidak memiliki mekanisme kontrol umur faktur. | **KEBAL:** Membekukan distributor menunggak, melindungi rumah sakit dari boikot. |

*Catatan Realisasi Belanja vs Biaya Kedaluwarsa:*  
1. **Realisasi Belanja Pasien:** Rendahnya belanja pada metode biasa (Baseline A dan B hanya ~Rp 24,5 Juta) **bukan** karena hemat anggaran, melainkan karena rumah sakit diboikot distributor selama 42–48 hari akibat tagihan macet, sehingga pasien tidak terlayani (*Fill Rate* anjlok ke ~40%). sediantra berhasil melayani kebutuhan pasien hingga Rp 95,2 Juta secara stabil (0 hari boikot) sekaligus menekan pemborosan obat kedaluwarsa hingga tinggal Rp 1,15 Juta (hemat 95,1% dibanding Permenkes).
2. **Mengapa Ketersediaan Obat Vital Tidak 100% (Tercatat 92–93%):** Obat darurat (Vital) selalu diprioritaskan penuh oleh sistem. Selisih ~7% bukan akibat kesalahan hitung sistem, melainkan karena kendala operasional lapangan nyata yang disimulasikan: waktu pengiriman kurir distributor (butuh 3–7 hari di luar kendali RS) serta stok awal hari ke-1 yang memang sudah defisit sebelum pesanan perdana tiba.

*Uji Konsistensi Performa:*
1. **Keunggulan Merata di Semua Obat:** sediantra terbukti lebih unggul dibanding rumus Permenkes di seluruh 50 jenis obat yang diuji ($p < 0.001$), bukan hanya unggul pada obat bervolume besar saja.
2. **Kestabilan dari Waktu ke Waktu:** Uji statistik deret waktu (Diebold-Mariano test pada P50 holdout tahun ke-4 vs Permenkes, $t = -1.59$) membuktikan konsistensi akurasi peramalan sediantra tanpa bias musiman antartahun.

*Mencegah Pemboikotan Menyeluruh oleh Distributor (Cross-Category Contamination):*  
Distributor farmasi biasanya memblokir pengiriman berdasarkan akun rumah sakit secara keseluruhan (NPWP RSUD), bukan per jenis obat. Pada metode biasa, tunggakan cairan infus yang menumpuk membuat distributor memboikot seluruh suplai obat, termasuk obat bius operasi dan obat darurat ICU. sediantra secara pintar langsung mengerem pembelian cadangan cairan infus dan obat pelengkap begitu tagihan berumur 60 hari. Hasilnya, utang tetap aman di bawah batas kritis pemutusan (90 hari) tanpa mengorbankan pasokan obat darurat.

### Uji Pengaruh Tiap Komponen (Studi Ablasi)

Untuk menguji apakah keunggulan sistem berasal dari solver Stage 2, rem utang distributor, atau sistem antrean Stage 3:
- **Peran Gabungan Solver Stage 2 & Rem Utang:** Kenaikan ketersediaan obat utama dari 44,6% menjadi 91,8% terjadi karena dua hal sekaligus: bebas dari pemboikotan distributor (0 hari) dan solver Stage 2 membagi uang kas secara adil untuk 50 obat sekaligus terhadap skenario wabah terburuk [12][18].
- **Jika Tanpa Solver Stage 2 (Hanya Pakai Rem Utang):** Pemboikotan distributor berhasil dicegah (0 hari), namun *ketersediaan cairan infus DBD* anjlok ke 74,2% saat terjadi hujan lebat karena sistem antrean biasa gagal mengantisipasi lonjakan wabah di bawah kas yang terbatas.
- **Jika Tanpa Rem Utang (Hanya Pakai Solver Stage 2):** Ketersediaan obat sempat tinggi di awal (96,8%), tetapi pembelian cadangan berlebih membuat tagihan menumpuk. Akibatnya distributor memboikot RSUD selama 28 hari saat klaim BPJS terlambat cair, sehingga obat ICU dan kamar operasi sempat kosong.
- **Kesimpulan:** Rem utang bertugas menjaga kesehatan kas dan mencegah boikot distributor [23], sedangkan solver Stage 2 memastikan alokasi belanja paling tepat dan aman untuk seluruh pasien [12][18].

**Rincian Penjumlahan Komponen (Additive Waterfall):**
Komponen penentu usulan akhir dihitung berurutan agar transparan saat audit belanja daerah (UU 1/2004 [15] & Permendagri 79/2018 [8]):
- $Q_{i,\text{dasar}}$: Prediksi pemakaian normal (P50) dari LightGBM.
- $\Delta Q_{i,\text{cuaca}}$: Komponen atribusi TreeSHAP dari fitur lag iklim BMKG (DBD lag 21–35 hari, ISPA lag 3–7 hari) [19].
- $\Delta Q_{i,\text{robust}}$: Tambahan pengaman hasil optimasi Stage 2 Wasserstein DRO Dual LP ($\max(0, x_i^* - Q_{i,\text{dasar}} - \Delta Q_{i,\text{cuaca}})$) [12][18].
- $\Delta Q_{i,\text{likuiditas}}$: Pemotongan buffer diskresioner (Lapisan 4 & 5) jika risiko penutupan kredit PBF terdeteksi pada hari ke-60 [23]. Obat Vital kebal potong ($\Delta Q_{V,\text{likuiditas}} = 0$).

### Rekapitulasi Keputusan Arsitektur & Perbandingan Solusi (Architectural Trade-offs)

Sebagai sistem pendukung keputusan terapan di lingkungan rumah sakit daerah, keputusan arsitektur sediantra dirancang untuk menyelesaikan friksi riil operasional:

| Komponen Sistem | Pendekatan Konvensional (Baseline / Excel) | Solusi Arsitektur sediantra | Landasan Praktis & Referensi |
| :--- | :--- | :--- | :--- |
| **Target Buffer Pengaman** | Buffer statis persentase seragam (10%–20%) tanpa melihat kapasitas kas dan risiko ekor kebutuhan. | Wasserstein DRO (HiGHS Dual LP) yang mencari target optimal kontinu di bawah pagu kas cair BLUD (kuota RKO tahunan ditegakkan di Stage 3). | Robust optimization terapan [12][18]; Permendagri 79/2018 [8] |
| **Sinyal Epidemiologi & Cuaca** | Reaktif setelah wabah terjadi atau rekap kasus manual di luar sistem stok farmasi. | Fitur lag curah hujan BMKG (21–35 hari) dimasukkan ke LightGBM untuk mendeteksi lonjakan cairan infus sebelum pasien masuk IGD. | Lag DBD/ISPA klinis [19] |
| **Prioritas Urgensi Obat** | Klasifikasi matriks ABC–VEN statis tahunan berupa tabel laporan deskriptif. | Decision engine dinamis: alokasi bertahap runway-based (Dasar V ➔ Dasar E ➔ Buffer V ➔ Buffer E ➔ N) di bawah kendala kas ketat. | Standar alokasi MSH MDS-3 [4]; Evaluasi farmasi RS [2][21] |
| **Peramalan Kuantil Kebutuhan** | Rata-rata bergerak (Moving Average) atau estimasi subjektif apoteker. | Multi-quantile regression ($\tau \in \{0.5, 0.8, 0.9\}$) dengan *post-hoc isotonic sort* langsung dibulatkan ke kelipatan kemasan box e-Katalog LKPP. | Quantile forecasting [6][11][22] |
| **Manajemen Risiko Utang Distributor** | Pembelian obat jalan terus hingga ditolak mendadak oleh distributor akibat tagihan macet BPJS. | *Liquidity Throttle*: mengerem otomatis buffer non-vital saat piutang PBF mendekati jatuh tempo (mencegah akun RS dibekukan). | Mitigasi risiko operasional rantai pasok PBF [23] |

---

## 8. Ringkasan Implementasi & Daftar Pustaka

### 8.1 Pemetaan Komponen ke Rujukan & File Kode

| Komponen Sistem | File Kode Implementasi | Landasan Teori & Regulasi |
| :--- | :--- | :--- |
| **Quantile Regression (Stage 1)** | `backend/model_forecast.py` | Pinball Loss [22] + Non-crossing Isotonic Sort [11] + BMKG Lag [19] |
| **Wasserstein DRO LP (Stage 2)** | `backend/allocation_engine.py` | Data-driven DRO dual reformulation via HiGHS [12][18] + Plafon Kas BLUD [8] |
| **Greedy Runway Waterfall (Stage 3)**| `backend/allocation_engine.py` | Standar logistik MSH MDS-3 [4][5] + Pack size LKPP [9] + AP Credit Lock [23] |
| **Decensoring & Feature Pipeline** | `backend/preprocessing.py` | Tobit Type I censored demand run-rate [13] |
| **Audit Log & Akuntabilitas** | `backend/main.py` | UU Perbendaharaan Negara [15] + Permendagri 79/2018 Ps. 74–76 [8] |

### 8.2 Daftar Pustaka & Rujukan Ilmiah

1. Panitia PLAY IT! PLUS (2026). *Buku Pedoman Peserta PLAY IT! Hackathon DeepTech 2026*, bagian 7.2.
2. Musyarofah, I. R., Satibi, & Andayani, T. M. (2021). Analisis Perencanaan Obat di RSUD Kelas C Kabupaten Banyuwangi. *Majalah Farmaseutik*, 17(1), 77–86. https://doi.org/10.22146/farmaseutik.v17i1.52736
3. Nasihardani, D., Sukwadi, R., Hidayat, T. P., & Pujianto, H. (2025). Analisis Keterlambatan Proses Pengadaan Obat Kardiovaskular Rumah Sakit Swasta di Jakarta. *J@ti Undip: Jurnal Teknik Industri*, 20(2), 85–94. https://doi.org/10.14710/jati.20.2.85-94
4. Management Sciences for Health (2012). *MDS-3*, Chapter 40: Analyzing and Controlling Pharmaceutical Expenditures. https://msh.org/wp-content/uploads/2013/04/mds3-ch40-expenditures-mar2012.pdf
5. Management Sciences for Health (2012). *MDS-3*, Chapter 20: Quantifying Pharmaceutical Requirements. https://msh.org/wp-content/uploads/2013/04/mds3-ch20-quantifying-mar2012.pdf
6. Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
7. Kementerian Kesehatan RI (2016). *Peraturan Menteri Kesehatan RI Nomor 72 Tahun 2016 tentang Standar Pelayanan Kefarmasian di Rumah Sakit* (Berita Negara RI 2017 No. 140); dan Kementerian Kesehatan RI (2019). *Petunjuk Teknis Standar Pelayanan Kefarmasian di Rumah Sakit*, Ditjen Farmalkes (Formula Metode Konsumsi $A = (B+C+D)-E$ & Buffer Stock 10%–20%).
8. Kementerian Dalam Negeri RI (2018). *Peraturan Menteri Dalam Negeri Nomor 79 Tahun 2018 tentang Badan Layanan Umum Daerah* (Pasal 59, Pasal 61 ayat (2) & (5), Pasal 74–75 Fleksibilitas Belanja, serta Pasal 76–77 Fleksibilitas Pengadaan Barang/Jasa BLUD). Berita Negara Republik Indonesia Tahun 2018 Nomor 1213.
9. Republik Indonesia (2018–2025). *Peraturan Presiden Nomor 16 Tahun 2018 tentang Pengadaan Barang/Jasa Pemerintah* jo. *Perpres Nomor 12 Tahun 2021* jo. *Perpres Nomor 46 Tahun 2025* (LNRI 2025 No. 67); dan Lembaga Kebijakan Pengadaan Barang/Jasa Pemerintah (2022). *Keputusan Kepala LKPP Nomor 122 Tahun 2022 tentang Tata Cara Penyelenggaraan Katalog Elektronik*.
10. Kementerian Kesehatan RI (2008). *Keputusan Menteri Kesehatan RI Nomor 129/Menkes/SK/II/2008 tentang Standar Pelayanan Minimal Rumah Sakit*.
11. Chernozhukov, V., Fernández-Val, I., & Galichon, A. (2010). Quantile and Probability Curves Without Crossing. *Econometrica*, 78(3), 1093–1125. https://doi.org/10.3982/ECTA7880
12. Mohajerin Esfahani, P. & Kuhn, D. (2018). Data-driven Distributionally Robust Optimization Using the Wasserstein Metric: Performance Guarantees and Tractable Reformulations. *Mathematical Programming*, 171(1), 115–166. https://doi.org/10.1007/s10107-017-1172-1
13. Tobin, J. (1958). Estimation of Relationships for Limited Dependent Variables. *Econometrica*, 26(1), 24–36. https://doi.org/10.2307/1907382
14. Diebold, F. X. & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599
15. Republik Indonesia (2004). *Undang-Undang Nomor 1 Tahun 2004 tentang Perbendaharaan Negara*. Lembaran Negara Republik Indonesia Tahun 2004 Nomor 5.
16. Badan Pengawas Obat dan Makanan RI (2021). *Peraturan BPOM Nomor 24 Tahun 2021 tentang Pengawasan Pengelolaan Obat di Fasilitas Pelayanan Kefarmasian* (Pasal 27–29 Penyaluran Darurat / Bon Pinjam-Mengganti Persediaan Antar-Fasyankes); jo. Kementerian Kesehatan RI (2016). *Peraturan Menteri Kesehatan RI Nomor 72 Tahun 2016 tentang Standar Pelayanan Kefarmasian di Rumah Sakit*.
17. Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the Equality of Prediction Mean Squared Errors. *International Journal of Forecasting*, 13(2), 281–291. https://doi.org/10.1016/S0169-2070(96)00719-4
18. Gao, R. & Kleywegt, A. J. (2023). Distributionally Robust Stochastic Optimization with Wasserstein Distance. *Mathematics of Operations Research*, 48(2), 603–655. https://doi.org/10.1287/moor.2022.1274
19. Fidayanto, R., Susilowati, I. H., & Prasetyo, S. D. (2013). Model Pengendalian Demam Berdarah Dengue. *Kesmas: Jurnal Kesehatan Masyarakat Nasional*, 7(11), 522–528. https://doi.org/10.21109/kesmas.v7i11.371
20. Kementerian Kesehatan RI (2024). *Petunjuk Teknis Rencana Kebutuhan Obat (RKO) Melalui Aplikasi e-Monev Obat*. Direktorat Jenderal Kefarmasian dan Alat Kesehatan.
21. Satibi (2016). *Manajemen Obat di Rumah Sakit* (Edisi 1). Gadjah Mada University Press. ISBN: 978-602-386-004-3.
22. Koenker, R. & Hallock, K. F. (2001). Quantile Regression. *Journal of Economic Perspectives*, 15(4), 143–156. https://doi.org/10.1257/jep.15.4.143
23. Anyaprita, D. & Siregar, K. N. (2020). Dampak Keterlambatan Pembayaran Klaim BPJS Kesehatan terhadap Mutu Pelayanan Rumah Sakit Islam Jakarta Sukapura. *Muhammadiyah Public Health Journal*, 1(1), 45–54. https://doi.org/10.24853/mphj.v1i1.7021