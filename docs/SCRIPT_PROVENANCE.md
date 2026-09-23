# Script provenance

Every published script under [`experiments/generalized_safety/`](../experiments/generalized_safety/) is a copy of the frozen cluster original. The only change allowed was replacing machine-specific paths.

## Sanitization rule

Hard-coded cluster paths were replaced with environment-variable lookups:

| Environment variable | Replaces |
|---|---|
| `CDG_WORK_ROOT` | the private working root that holds the frozen `analysis_output/rrae_development/generalized_safety_v2_*` artifact tree |
| `CDG_LLADA_MODEL` | local LLaDA-8B-Instruct weights directory |
| `CDG_DIJA_ROOT` | local DIJA code checkout |

Each substitution is a single-line, in-place expression (`__import__("os").environ[...]`), so line numbers match the originals. For every file, applying the inverse substitution reproduces the frozen original **byte for byte**. That is the check behind `SCIENTIFIC_LOGIC_CHANGED=false`: no equation, seed, threshold, label, dataset, metric, retry rule, acceptance rule or generation parameter changed.

**Runtime SHA pins.** Several scripts SHA-verify their inputs and the other scripts they import (for example, `m2_llada_runtime_v1.EXPECT` and `EXPECTED_*_SHA` in the runner). Those pins name the **original** frozen files and were deliberately left unchanged. Fail-closed execution therefore requires the original frozen sources, identified by `ORIGINAL_CLUSTER_SHA` below. The sanitized copies are published for inspection and provenance. Pointing them at private inputs without the originals makes them stop at the SHA check, by design.

The private inputs (datasets, manifests, private maps, judgments, generations, hidden states, directions) are not published, so no stage can be re-executed from this repository alone.

## Mapping

| Stage | Published path | Role | ORIGINAL_CLUSTER_SHA | PUBLISHED_SANITIZED_SHA | Sanitized lines | SCIENTIFIC_LOGIC_CHANGED |
|---|---|---|---|---|---|---|
| shared | `shared/run_generalized_safety_v2_l16_full_extraction_v1.py` | L16 fully-visible response extraction (V2 direction inputs) | `337667529a9da58a283fc0cad56dd74412e4bd637ff1602e3e65f5f065d462f0` | `8fafe97cddce44d3497f0937adcdba2f88bb46303c266158e7116030ca6405d5` | 15, 17 | false |
| shared | `shared/run_generalized_safety_m1_eval_v1.py` | LLaDA steered-generation runner (M1_PATH; used by M1 and M2) | `3ff21e59c7d045c42187d9880410280e92d2efcd6a585ecad1db984b63da618f` | `95876e367d4533638f984de99447b0913909a6b689ca3a8ce69eb6beff03282f` | 19, 22 | false |
| shared | `shared/run_generalized_safety_direction_construction_baselines_v2.py` | baseline helper imported by runner (handoff §57 mislabels it 'direction construction script') | `fa5ad065c0c1e8b69228bc12356e8ee1cdb10c38ea864d2e21f3d8bdabbfae62` | `3f524d088c3e752ed209eaa9ad515e93793dbb2c335849febb5c0bd33d95f135` | 21, 23 | false |
| shared | `shared/run_canonical_train_layer_matched_safety_dose_response_canary_v1.py` | historical dose-response canary module imported by runner (HIST_PATH) | `cca5bb26111932c78815ef5a75bdc4898ce9a1a023550c1b9a06dbcbacdae224` | `cca5bb26111932c78815ef5a75bdc4898ce9a1a023550c1b9a06dbcbacdae224` | — (unchanged; SHA identical) | false |
| shared | `shared/extract_iterative_hidden_canonical_dija_v1.py` | canonical DIJA region/mask helper (HELPER_PATH) | `67832c5cb193699b2fe83b545b6013b290dc8cddbd8d35504c983bda1cb92680` | `67832c5cb193699b2fe83b545b6013b290dc8cddbd8d35504c983bda1cb92680` | — (unchanged; SHA identical) | false |
| shared | `shared/frozen_generation_helpers_compat_v1.py` | frozen DIJA localization compat helper | `98189fd24be8789891507c19ab1d9f630b5ec5dd5c85e74b8b76b97ccdae577a` | `98189fd24be8789891507c19ab1d9f630b5ec5dd5c85e74b8b76b97ccdae577a` | — (unchanged; SHA identical) | false |
| shared | `shared/run_m2_dija_construction_v2.py` | DIJA attack construction executor (M2 confirmatory, reused by M3') | `d2671c55759a7483fb390953dc42c83f2a962c907ae6ea42b4919af27b1aaaaf` | `a3a6aa764a2091e39401a6713361a7fb07edd6b517487ae06f7a6c53d2740c94` | 31 | false |
| shared | `shared/m2_dija_seed_policy_v2.py` | DIJA seed policy | `0234fafc0697e8aa20e269c15d4d354c6c809bd06ac525f2092d6e1897f6e98b` | `0234fafc0697e8aa20e269c15d4d354c6c809bd06ac525f2092d6e1897f6e98b` | — (unchanged; SHA identical) | false |
| shared | `shared/run_m2_renellm_construction_v1.py` | ReNeLLM attack construction executor (M2; M3' binding UNRESOLVED) | `7973e74516b07fd7c39df4c8b41e6786da820fa9b3a1d72563bcea5cc7625673` | `0b1fe6a54e7b0252d4dbf27c681f2070233c7af32d58405fa96dd9a4fefc6052` | 21 | false |
| M1 | `m1/run_m1_heldout_shard_v1.py` | held-out generation shard executor | `d6ca9d6641cf0967f821b976d6f76627675d4c9192524a42775070c3de09b52f` | `7c76641a63614af512499ea50e837543a95ac28a7393002ceb514f2ed496c52f` | 16, 17 | false |
| M1 | `m1/run_heldout_m1_deepseek_judging_final.py` | held-out DeepSeek judging executor (bound by judging authorization) | `8fcbaf043385def6b963f9b6d842c66c954515969d4a7c8d7ead638b02ca8c35` | `8fcbaf043385def6b963f9b6d842c66c954515969d4a7c8d7ead638b02ca8c35` | — (unchanged; SHA identical) | false |
| M1 | `m1/run_heldout_m1_unblinding_metrics_v1.py` | held-out unblinding + frozen metrics executor | `cf89032fdd7057601bf5f59c7cf0a2c40afe8d1f39dcb3a7ae8b9c5f356d778a` | `cf89032fdd7057601bf5f59c7cf0a2c40afe8d1f39dcb3a7ae8b9c5f356d778a` | — (unchanged; SHA identical) | false |
| M2 | `m2/m2_norm_preserving_hook_v1.py` | norm-preserving hook | `20370c45607b8acea040bc1aa5a1b3355d301a7d0eeadfdadcadfce0c48e49d1` | `20370c45607b8acea040bc1aa5a1b3355d301a7d0eeadfdadcadfce0c48e49d1` | — (unchanged; SHA identical) | false |
| M2 | `m2/build_contract.py` | contract builder | `9e2daaad2523b649765b6c94cf0e9b9c70a3c9e07998287125ee4eb33d1e0567` | `5d350224415d6b811d7c67a71f9ca29e8a21472f8a3416856cd142bf869ca135` | 5, 163 | false |
| M2 | `m2/preflight_m2_measurement_v1.py` | calibration measurement preflight | `24bf0c2fc4112fc629adf4b971d37d4042fac86cd79f41f832d1236ae51a4522` | `4ec9e7d8e111130a1c7468875d66fe9e912e7b788bc3adf871487efcdee71a30` | 13 | false |
| M2 | `m2/run_m2_measurement_shard_v1.py` | calibration measurement shard executor | `d3f075ba1718216935c6e75d5db933d2d637cbd1c5cfdddf608320afbdb9e02c` | `908d24402607f4f0d86e77bc2077141247a21b98f511d4a8411422984d17e842` | 24, 26 | false |
| M2 | `m2/solve_m2_betas_v1.py` | displacement-matched beta solver | `9bea814369d16c7bb2d79d12748f5f5a04be65743442395e4e8961ea1bacb325` | `5873f1db11b5990d02e539a6e859465ca41d5c6b19eb6ac3c021ea6acf384c6e` | 17 | false |
| M2 | `m2/m2_llada_runtime_v1.py` | generation runtime | `5425cd6466979c6cf7c7012e2d94063000a5695494efc4a1cd4b6a365719cb0b` | `ebb919e9c0165989448779ed94e0529a7d19bd0fb37ca41fd204a58ed8f95115` | 17, 20 | false |
| M2 | `m2/generate_one_m2dmnp_source.py` | patched generate (DMNP source) | `45693f610249d02a8336957f8088d2f83b91e796e1266c47accb740702a3bad2` | `45693f610249d02a8336957f8088d2f83b91e796e1266c47accb740702a3bad2` | — (unchanged; SHA identical) | false |
| M2 | `m2/build_manifests_v1.py` | manifest builder / route_and_input | `3a6e95b5e373e31b382f6ccc4e73d50a11fc5191c88aa07944f6af7020e93cf1` | `3a6e95b5e373e31b382f6ccc4e73d50a11fc5191c88aa07944f6af7020e93cf1` | — (unchanged; SHA identical) | false |
| M2 | `m2/run_m2_llada_confirmatory_shard_v1.py` | confirmatory generation shard executor | `2f48bdfdf15db41887d2704c13e3bc22bdbabfba22f2789407b9c3bc905c1d1a` | `2f48bdfdf15db41887d2704c13e3bc22bdbabfba22f2789407b9c3bc905c1d1a` | — (unchanged; SHA identical) | false |
| M2 | `m2/run_m2_fresh_deepseek_judging.py` | fresh confirmatory DeepSeek judging executor | `36328fd1ed6282bd07df2276021a19395b1cbc48c6ee9b40f6629ec056b01699` | `36328fd1ed6282bd07df2276021a19395b1cbc48c6ee9b40f6629ec056b01699` | — (unchanged; SHA identical) | false |
| M2 | `m2/02_build_packet_and_private_map.py` | judging packet + private map builder | `d48674a854a62798f104d968b23eec8bed28915d7e1fb4104aed92c6103ed015` | `bd65748ca9de9fa103ff110489bcff1eeb29bbf63c94bb6b9ba0dce414963a9e` | 7, 29 | false |
| M2 | `m2/07_post_judging_audit_and_freeze.py` | post-judging audit + freeze | `4247b2afbd627a53268c9ed6ab0d35e9224838d4fa6bc7b7f3d8dd4a3b457922` | `4247b2afbd627a53268c9ed6ab0d35e9224838d4fa6bc7b7f3d8dd4a3b457922` | — (unchanged; SHA identical) | false |
| M2 | `m2/run_m2_unblinding_metrics_v1.py` | unblinding + frozen metrics executor | `557e757ff5ee13009f88eea13a692b1cfbf9d048aff60ad56cf0152a8aa4bd5b` | `557e757ff5ee13009f88eea13a692b1cfbf9d048aff60ad56cf0152a8aa4bd5b` | — (unchanged; SHA identical) | false |
| M3 | `m3/tgas_feasibility_spike_v1.py` | TGAS feasibility spike | `f15d5e2024e74739ab776e203035365b478ba365d7c5e1113baad2e8c016cf9a` | `c972345a4603dc3916c8a9ee8c96876de4b651a65a44a2903192bb95db50df5f` | 313, 323 | false |
| M3 | `m3/m3_power_simulation_v1.py` | power simulation | `cedf084a8322f93c7f565a9277e819fa295adfc392c468118ce8d8118ca7f590` | `cedf084a8322f93c7f565a9277e819fa295adfc392c468118ce8d8118ca7f590` | — (unchanged; SHA identical) | false |
| M3 | `m3/scenario_match_audit.py` | scenario-match audit (fail-closed) | `d4dd839998a65eda767a2d0e445c3dbee91a6dc543edf22227b66974c6c22055` | `d4dd839998a65eda767a2d0e445c3dbee91a6dc543edf22227b66974c6c22055` | — (unchanged; SHA identical) | false |
| M3 | `m3/validate_against_known_cases.py` | scenario-match audit regression validation | `a27bcfe50ca47d016171e1d70b537512d6161c0896b3a37e27e2d7dff039c076` | `57df136dbceb41131969000994d3d9bef0b675d3f14e038ee31f17a7005ac20c` | 15 | false |
| M3 | `m3/01_rebuild_packet_v2.py` | BUILD/DEV judging packet rebuild v2 | `ddec023031ca4eb36dbd30c4ecc4558957fcbdf7e9fb2cb0b5040b398ca06623` | `b87b5e3eff89faf9b0c21879aa9f365eb4dddc622986abf16cb22ed423a9023a` | 38, 42 | false |
| M3 | `m3/02_run_judging_v2.py` | BUILD/DEV judging run v2 | `371348c9195ecbc450041f3775bde60cbdd98aff69a12d07ffa720875dae453e` | `371348c9195ecbc450041f3775bde60cbdd98aff69a12d07ffa720875dae453e` | — (unchanged; SHA identical) | false |
| M3 | `m3/03_integrity_audit_and_freeze_v2.py` | BUILD/DEV integrity audit + freeze v2 | `9e245f8b9a967e007acad9317a4ef6703a433b3b2d6c80e6e6692f350a7128c8` | `9e245f8b9a967e007acad9317a4ef6703a433b3b2d6c80e6e6692f350a7128c8` | — (unchanged; SHA identical) | false |
| M3 | `m3/04_unblind_dev_support_check_v2.py` | DEV support check (final M3 decision) | `e84e5f4e891963f785b20aade8283fbb1fdde997298303932d00e2aa1f709cfa` | `e84e5f4e891963f785b20aade8283fbb1fdde997298303932d00e2aa1f709cfa` | — (unchanged; SHA identical) | false |
| M3' | `m3prime/m3prime_llada_generate.py` | NEW_DEV baseline generation + L16 trajectory observer (NewDevFeatureObserver) | `c90d968a105b6059f425b3372ad47f389a12f70f43b723293dc781892fff01e2` | `9a7b6bb3e589ee28fbb8f2455e774e0b195ac1f115aeb38fb755b98209f194f0` | 29 | false |
| M3' | `m3prime/05_freeze_generation.py` | NEW_DEV generation freeze | `d26fee6527919cafb55393b22ee93675718c9345584dd88c9d558e86bc6bdfa3` | `d26fee6527919cafb55393b22ee93675718c9345584dd88c9d558e86bc6bdfa3` | — (unchanged; SHA identical) | false |
| M3' | `m3prime/01_build_packet.py` | NEW_DEV judging packet builder | `ca42092c07ff945c68a0627770c886f8eca2aab207e01e39a9a99e5c0ee277ab` | `2212bf71da9d6f5f3de8c3a8378bd6e394616cf7715a12f5d5206a752e2f0ca2` | 28 | false |
| M3' | `m3prime/02_run_judging.py` | NEW_DEV judging run | `5d8a0aa068722ab5d95e3eea891f54924e7bb1f9f235463cf736560cd768ea6a` | `5d8a0aa068722ab5d95e3eea891f54924e7bb1f9f235463cf736560cd768ea6a` | — (unchanged; SHA identical) | false |
| M3' | `m3prime/03_integrity_audit_and_freeze.py` | NEW_DEV integrity audit + freeze | `5ac7de347f13aa56e41f5d1a1a9353634f5b6aef0640e60119656dfe5ec8a6e7` | `5ac7de347f13aa56e41f5d1a1a9353634f5b6aef0640e60119656dfe5ec8a6e7` | — (unchanged; SHA identical) | false |
| M3' | `m3prime/04_support_check.py` | NEW_DEV support check | `5ba4055af1ce7888c5491ce562de0ed5bf2b9e512265fba5c8ffea439a2b77ea` | `5ba4055af1ce7888c5491ce562de0ed5bf2b9e512265fba5c8ffea439a2b77ea` | — (unchanged; SHA identical) | false |
| M3' | `m3prime/06_build_capture.py` | BUILD trajectory capture (step 8) | `fd696629e08199fee96a385d115203371f159b4a30d2c439677acb5f8c49373a` | `02f58f36a48e3120a7324d92e6017e6516d1e07f356f3a06971eef2a4fd6e953` | 17 | false |
| M3' | `m3prime/07_byte_identity_and_freeze.py` | BUILD byte-identity + feature freeze | `34753700df0499ca6e3c7276a645557ce56c1b6a5d047d8ad041dc6a66e9c94f` | `676399c6317986bede6850892f47c3c6e6b210b7176643e5980be6d7ce56688e` | 23 | false |
| M3' | `m3prime/08_gate_fit.py` | gate training (BUILD, l2 logistic) | `b991e0dc6a6a2a86599ec8f75ababbbee781cc7033b3bb0eff04165e27b968d0` | `cb8858ac06c2c2a3725efea72a9255cad576fe802c77abaafad956ec31539508` | 24 | false |
| M3' | `m3prime/09_gate_qualification.py` | threshold qualification (NEW_DEV) | `3d18ac171fab2210708f48e2d6af627ef0f6bf0bbe3fd713d9e4442d5fda158c` | `7d61345d9f2d2cfc3de12abf3572156739f7556558a05544726db80ef3c99dcf` | 24 | false |
| M3' | `m3prime/build_final_result.py` | final result builder | `51b18295ad9c4916c75b9ad3bb4b70c66a4b6e8306cc3c460c50835ce7f409f1` | `4e717fc65924d5ee96188f1ea8a04899f558cfe6936daecc37d4a0fbd1e977fb` | 6 | false |

Original cluster location: the `source_path` column of the release inventory. Paths are relative to the private working root:

| Published path | Original path (relative to `CDG_WORK_ROOT`) |
|---|---|
| `shared/run_generalized_safety_v2_l16_full_extraction_v1.py` | `scripts/rrae_development/run_generalized_safety_v2_l16_full_extraction_v1.py` |
| `shared/run_generalized_safety_m1_eval_v1.py` | `scripts/rrae_development/run_generalized_safety_m1_eval_v1.py` |
| `shared/run_generalized_safety_direction_construction_baselines_v2.py` | `scripts/rrae_development/run_generalized_safety_direction_construction_baselines_v2.py` |
| `shared/run_canonical_train_layer_matched_safety_dose_response_canary_v1.py` | `scripts/official_rrae/run_canonical_train_layer_matched_safety_dose_response_canary_v1.py` |
| `shared/extract_iterative_hidden_canonical_dija_v1.py` | `scripts/official_rrae/extract_iterative_hidden_canonical_dija_v1.py` |
| `shared/frozen_generation_helpers_compat_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_llada_baseline_dija_localization_compat_v1/frozen_generation_helpers_compat_v1.py` |
| `shared/run_m2_dija_construction_v2.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_dija_construction_protocol_v2/run_m2_dija_construction_v2.py` |
| `shared/m2_dija_seed_policy_v2.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_dija_construction_protocol_v2/m2_dija_seed_policy_v2.py` |
| `shared/run_m2_renellm_construction_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_renellm_execution_v1/run_m2_renellm_construction_v1.py` |
| `m1/run_m1_heldout_shard_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m1_heldout_llada_execution_v1/run_m1_heldout_shard_v1.py` |
| `m1/run_heldout_m1_deepseek_judging_final.py` | `analysis_output/rrae_development/generalized_safety_v2_m1_heldout_judge_executor_final_adaptation_v2/run_heldout_m1_deepseek_judging_final.py` |
| `m1/run_heldout_m1_unblinding_metrics_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m1_heldout_unblinding_executor_adaptation_v1/run_heldout_m1_unblinding_metrics_v1.py` |
| `m2/m2_norm_preserving_hook_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_contract_v1/m2_norm_preserving_hook_v1.py` |
| `m2/build_contract.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_contract_v1/build_contract.py` |
| `m2/preflight_m2_measurement_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_beta_calibration_measurement_v1/preflight_m2_measurement_v1.py` |
| `m2/run_m2_measurement_shard_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_beta_calibration_measurement_v1/run_m2_measurement_shard_v1.py` |
| `m2/solve_m2_betas_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_beta_calibration_measurement_v1/solve_m2_betas_v1.py` |
| `m2/m2_llada_runtime_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_llada_executor_v1/m2_llada_runtime_v1.py` |
| `m2/generate_one_m2dmnp_source.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_llada_executor_v1/generate_one_m2dmnp_source.py` |
| `m2/build_manifests_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_llada_executor_v1/build_manifests_v1.py` |
| `m2/run_m2_llada_confirmatory_shard_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_llada_executor_v1/run_m2_llada_confirmatory_shard_v1.py` |
| `m2/run_m2_fresh_deepseek_judging.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_fresh_confirmatory_judging_v1/run_m2_fresh_deepseek_judging.py` |
| `m2/02_build_packet_and_private_map.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_fresh_confirmatory_judging_v1/02_build_packet_and_private_map.py` |
| `m2/07_post_judging_audit_and_freeze.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_fresh_confirmatory_judging_v1/07_post_judging_audit_and_freeze.py` |
| `m2/run_m2_unblinding_metrics_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m2_unblinding_executor_preflight_v1/run_m2_unblinding_metrics_v1.py` |
| `m3/tgas_feasibility_spike_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m3_feasibility_spike_v1/tgas_feasibility_spike_v1.py` |
| `m3/m3_power_simulation_v1.py` | `analysis_output/rrae_development/generalized_safety_v2_m3_power_calculation_v1/m3_power_simulation_v1.py` |
| `m3/scenario_match_audit.py` | `analysis_output/rrae_development/generalized_safety_v2_m3_scenario_match_audit_v1/scenario_match_audit.py` |
| `m3/validate_against_known_cases.py` | `analysis_output/rrae_development/generalized_safety_v2_m3_scenario_match_audit_v1/validate_against_known_cases.py` |
| `m3/01_rebuild_packet_v2.py` | `analysis_output/rrae_development/generalized_safety_v2_m3_build_dev_judging_v2/01_rebuild_packet_v2.py` |
| `m3/02_run_judging_v2.py` | `analysis_output/rrae_development/generalized_safety_v2_m3_build_dev_judging_v2/02_run_judging_v2.py` |
| `m3/03_integrity_audit_and_freeze_v2.py` | `analysis_output/rrae_development/generalized_safety_v2_m3_build_dev_judging_v2/03_integrity_audit_and_freeze_v2.py` |
| `m3/04_unblind_dev_support_check_v2.py` | `analysis_output/rrae_development/generalized_safety_v2_m3_build_dev_judging_v2/04_unblind_dev_support_check_v2.py` |
| `m3prime/m3prime_llada_generate.py` | `analysis_output/rrae_development/generalized_safety_v2_m3prime_llada_baseline_v1/m3prime_llada_generate.py` |
| `m3prime/05_freeze_generation.py` | `analysis_output/rrae_development/generalized_safety_v2_m3prime_llada_baseline_v1/05_freeze_generation.py` |
| `m3prime/01_build_packet.py` | `analysis_output/rrae_development/generalized_safety_v2_m3prime_new_dev_judging_v1/01_build_packet.py` |
| `m3prime/02_run_judging.py` | `analysis_output/rrae_development/generalized_safety_v2_m3prime_new_dev_judging_v1/02_run_judging.py` |
| `m3prime/03_integrity_audit_and_freeze.py` | `analysis_output/rrae_development/generalized_safety_v2_m3prime_new_dev_judging_v1/03_integrity_audit_and_freeze.py` |
| `m3prime/04_support_check.py` | `analysis_output/rrae_development/generalized_safety_v2_m3prime_new_dev_judging_v1/04_support_check.py` |
| `m3prime/06_build_capture.py` | `analysis_output/rrae_development/generalized_safety_v2_m3prime_build_features_v1/06_build_capture.py` |
| `m3prime/07_byte_identity_and_freeze.py` | `analysis_output/rrae_development/generalized_safety_v2_m3prime_build_features_v1/07_byte_identity_and_freeze.py` |
| `m3prime/08_gate_fit.py` | `analysis_output/rrae_development/generalized_safety_v2_m3prime_build_features_v1/08_gate_fit.py` |
| `m3prime/09_gate_qualification.py` | `analysis_output/rrae_development/generalized_safety_v2_m3prime_build_features_v1/09_gate_qualification.py` |
| `m3prime/build_final_result.py` | `analysis_output/rrae_development/generalized_safety_v2_m3prime_final_result_v1/build_final_result.py` |

## Withheld

- `analysis_output/rrae_development/generalized_safety_v2_m3_trajectory_features_v1/extract_trajectory_features_v1.py` (`b2b17054c53b33c5f39209275955235068b283a29d191552094df6544343c2b0`): **withheld**. Defective M3 trajectory extractor: it cannot reproduce BUILD inputs (no DIJA infill structure, no chat template) and was never executed on BUILD. M3′ used the observer in `m3prime/m3prime_llada_generate.py` instead.
- All `*.sbatch` / Slurm launch wrappers: **not published**. They are cluster-specific (account, GPU partition, absolute log and virtualenv paths) and contain no scientific logic; the Python entry points above carry all of it.

Totals: 44 published (22 sanitized, 22 byte-identical to the original), 1 withheld Python script, plus the Slurm wrappers.

## Missing originals

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md#provenance-gaps). No reconstruction is published, and nothing here is presented as an original executor unless it has a frozen SHA.
