# ERD Datamart Sediantara

Entity Relationship Diagram (ERD) dan Kontrak Skema Analitik untuk Replika Baca (*Read-Replica*) SIMRS & Data Mart Pengadaan Obat Sediantara.

```mermaid
erDiagram
    dim_date ||--o{ dim_hospital_context : "context_on"
    dim_date ||--o{ fct_prescription_dispensing : "tx_on"
    dim_date ||--o{ fct_inventory_snapshot : "snapshot_on"
    dim_date ||--o{ fct_purchase_orders : "ordered_on"
    dim_date ||--o{ fct_override_log : "override_on"

    dim_supplier ||--o{ dim_drug_master : "supplies"
    dim_supplier ||--o{ fct_purchase_orders : "billed_by"

    dim_ven_policy ||--o{ dim_drug_master : "governs"

    dim_drug_master ||--o{ fct_prescription_dispensing : "prescribed_in"
    dim_drug_master ||--o{ fct_inventory_snapshot : "stored_in"
    dim_drug_master ||--o{ fct_purchase_orders : "purchased_in"
    dim_drug_master ||--o{ fct_override_log : "adjusted_in"

    dim_location ||--o{ fct_prescription_dispensing : "dispensed_at"
    dim_location ||--o{ fct_inventory_snapshot : "counted_at"

    dim_budget_allocation ||--o{ fct_purchase_orders : "funded_by"
    dim_budget_allocation ||--o{ fct_override_log : "tracked_against"

    dim_date {
        date date_key PK
        int day_of_week
        boolean is_holiday
    }

    dim_supplier {
        varchar supplier_id PK
        varchar supplier_name
        varchar npwp
        int payment_terms_days
        decimal credit_limit_rupiah
        boolean is_active
    }

    dim_ven_policy {
        char ven_category PK "V, E, N"
        decimal allocation_weight
        decimal shortage_cost
        decimal excess_cost
    }

    dim_drug_master {
        varchar item_id PK
        varchar item_name
        varchar kfa_code
        varchar lkpp_commodity_id
        varchar nie_bpom
        varchar atc_code
        varchar therapeutic_interchange_group
        varchar uom
        boolean is_fornas
        boolean is_formularium_rs
        boolean is_high_alert
        char ven_category FK
        char abc_class "A, B, C"
        int pack_size
        decimal unit_price
        int min_order_qty
        int default_lead_time_days
        int vital_min_stock
        int annual_rko_quota_pack
        decimal holding_cost_rate
        varchar supplier_id FK
        boolean is_active
    }

    dim_location {
        varchar location_id PK
        varchar location_name
        varchar location_type "IGD, RJ, RI, GUDANG"
        boolean is_warehouse
        int internal_lead_time_days
    }

    dim_hospital_context {
        date tx_date PK, FK
        decimal inpatient_bor
        int outpatient_visits
        int emergency_visits
        decimal rainfall_mm
    }

    dim_budget_allocation {
        varchar budget_period PK "YYYY-MM"
        varchar budget_source PK "KAS_BLUD, APBD, DAK"
        decimal budget_limit
    }

    fct_prescription_dispensing {
        date tx_date PK, FK
        varchar item_id PK, FK
        varchar location_id PK, FK
        decimal qty_prescribed
        decimal qty_dispensed
        boolean is_unmet_or_substituted
        boolean is_stockout
    }

    fct_inventory_snapshot {
        date snapshot_date PK, FK
        varchar item_id PK, FK
        varchar location_id PK, FK
        varchar batch_number PK
        date expiry_date
        decimal current_stock
        decimal emergency_trolley_stock
        boolean is_quarantine
    }

    fct_purchase_orders {
        varchar po_id PK
        varchar item_id PK, FK
        varchar supplier_id FK
        date order_date FK
        date expected_date FK
        date received_date FK
        int order_qty_pack
        int received_qty_pack
        decimal unit_pack_price
        decimal total_line_rupiah
        varchar budget_period FK
        varchar budget_source FK
        varchar status "PENDING, ORDERED, RECEIVED, CANCELLED"
        date invoice_date
        date invoice_due_date
        varchar payment_status "PAID, UNPAID"
    }

    fct_override_log {
        varchar override_id PK
        varchar item_id FK
        varchar budget_period FK
        varchar budget_source FK
        date run_date FK
        varchar user_id
        varchar user_name
        varchar preset_mode
        int recommended_box
        int final_box
        text reason
        timestamp created_at
    }
```

---

### Aturan Relasi Data
1. **Multi-Source Funding Integrity:** `dim_budget_allocation` menggunakan composite primary key `(budget_period, budget_source)` agar pencatatan kas operasional BLUD, subsidi APBD, dan DAK dalam bulan yang sama tidak bertabrakan.
2. **AP Aging & Supplier Lock:** `fct_purchase_orders` menyimpan `invoice_due_date` dan `payment_status` yang berelasi dengan `dim_supplier.supplier_id` untuk menghitung umur utang invoice terlama ($\ge 60$ hari) secara otomatis.
3. **Audit Akuntabilitas BPK/Itjen:** Setiap penyesuaian (*override*) rekomendasi sistem wajib mencatat identitas apoteker (`user_id`, `user_name`), mode preset yang digunakan, dan stempel waktu UTC (`created_at`) di `fct_override_log`.
