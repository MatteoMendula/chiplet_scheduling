Network LLaVA_7B_EE_Correct {
	Layer PATCH_EMBED {
		Type: CONV
		Stride { X: 14, Y: 14 }
		Dimensions { K: 1024, C: 3, R: 14, S: 14, Y: 336, X: 336 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK01_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK01_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK01_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK01_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK01_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK01_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK02_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK02_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK02_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK02_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK02_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK02_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK03_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK03_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK03_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK03_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK03_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK03_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK04_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK04_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK04_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK04_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK04_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK04_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK05_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK05_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK05_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK05_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK05_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK05_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK06_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK06_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK06_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK06_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK06_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK06_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK07_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK07_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK07_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK07_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK07_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK07_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK08_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK08_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK08_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK08_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK08_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK08_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK09_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK09_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK09_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK09_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK09_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK09_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK10_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK10_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK10_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK10_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK10_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK10_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK11_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK11_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK11_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK11_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK11_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK11_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK12_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK12_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK12_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK12_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK12_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK12_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK13_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK13_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK13_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK13_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK13_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK13_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK14_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK14_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK14_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK14_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK14_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK14_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK15_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK15_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK15_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK15_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK15_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK15_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK16_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK16_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK16_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK16_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK16_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK16_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK17_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK17_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK17_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK17_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK17_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK17_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK18_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK18_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK18_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK18_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK18_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK18_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK19_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK19_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK19_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK19_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK19_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK19_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK20_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK20_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK20_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK20_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK20_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK20_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK21_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK21_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK21_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK21_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK21_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK21_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK22_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK22_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK22_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK22_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK22_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK22_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK23_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK23_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK23_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK23_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK23_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK23_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK24_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK24_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK24_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK24_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK24_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VIT_BLK24_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer VISION_TO_LLM_PROJ {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK01_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK01_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK01_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK01_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK01_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK01_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK01_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK02_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK02_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK02_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK02_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK02_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK02_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK02_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK03_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK03_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK03_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK03_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK03_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK03_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK03_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK04_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK04_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK04_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK04_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK04_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK04_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK04_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK05_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK05_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK05_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK05_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK05_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK05_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK05_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK06_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK06_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK06_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK06_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK06_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK06_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK06_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK07_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK07_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK07_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK07_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK07_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK07_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK07_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK08_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK08_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK08_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK08_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK08_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK08_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK08_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer EARLY_EXIT_HEAD_8 {
		Type: CONV
		Dimensions { K: 32000, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK09_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK09_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK09_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK09_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK09_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK09_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK09_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK10_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK10_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK10_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK10_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK10_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK10_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK10_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK11_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK11_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK11_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK11_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK11_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK11_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK11_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK12_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK12_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK12_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK12_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK12_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK12_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK12_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK13_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK13_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK13_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK13_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK13_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK13_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK13_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK14_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK14_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK14_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK14_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK14_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK14_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK14_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK15_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK15_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK15_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK15_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK15_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK15_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK15_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK16_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK16_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK16_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK16_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK16_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK16_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK16_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer EARLY_EXIT_HEAD_16 {
		Type: CONV
		Dimensions { K: 32000, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK17_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK17_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK17_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK17_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK17_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK17_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK17_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK18_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK18_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK18_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK18_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK18_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK18_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK18_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK19_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK19_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK19_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK19_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK19_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK19_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK19_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK20_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK20_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK20_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK20_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK20_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK20_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK20_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK21_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK21_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK21_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK21_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK21_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK21_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK21_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK22_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK22_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK22_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK22_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK22_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK22_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK22_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK23_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK23_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK23_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK23_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK23_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK23_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK23_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK24_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK24_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK24_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK24_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK24_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK24_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK24_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer EARLY_EXIT_HEAD_24 {
		Type: CONV
		Dimensions { K: 32000, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK25_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK25_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK25_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK25_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK25_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK25_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK25_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK26_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK26_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK26_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK26_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK26_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK26_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK26_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK27_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK27_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK27_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK27_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK27_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK27_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK27_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK28_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK28_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK28_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK28_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK28_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK28_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK28_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK29_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK29_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK29_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK29_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK29_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK29_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK29_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK30_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK30_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK30_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK30_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK30_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK30_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK30_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK31_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK31_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK31_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK31_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK31_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK31_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK31_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK32_ATTN_Q {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK32_ATTN_K {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK32_ATTN_V {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK32_ATTN_OUT {
		Type: CONV
		Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK32_MLP_GATE {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK32_MLP_UP {
		Type: CONV
		Dimensions { K: 11008, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer LLM_BLK32_MLP_DOWN {
		Type: CONV
		Dimensions { K: 4096, C: 11008, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

	Layer FINAL_LM_HEAD {
		Type: CONV
		Dimensions { K: 32000, C: 4096, R: 1, S: 1, Y: 1024, X: 1 }
                Dataflow {
                        // NVDLA-like dataflow
                        SpatialMap(1,1) K;
                        TemporalMap(64,64) C;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        Cluster(64, P);
                        SpatialMap(1,1) C;
                        TemporalMap(Sz(R),1) Y;
                        TemporalMap(Sz(S),1) X;
                        TemporalMap(Sz(R),Sz(R)) R;
                        TemporalMap(Sz(S),Sz(S)) S;
                }
	}

}
