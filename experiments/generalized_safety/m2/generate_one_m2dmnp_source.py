def generate_one_m2dmnp(
    *,
    item,
    model,
    tokenizer,
    helper,
    baseline,
    banks,
    device,
    step_audit_policy,
):
    set_seed(int(item["generation_seed"]))

    x_cpu, regions, metadata = build_input(
        item,
        tokenizer,
        helper,
        baseline,
    )

    x = x_cpu.to(device)
    initial_x = x.detach().clone()

    initial_mask = x == MASK_ID
    initial_count = int(initial_mask.sum().item())

    require(initial_count > 0, "No fillable masks")

    family = item["attack_family"]

    if family == "DIJA":
        trajectory_steps = initial_count
    else:
        require(initial_count == 128, "ReNeLLM initial mask count drift")
        trajectory_steps = 128

    for bank in banks.values():
        bank.reset_run(
            layer=None,
            beta=0.0,
            active_steps=[],
        )

    vector_key = item["vector_key"]

    selected_bank = None

    if vector_key is not None:
        selected_bank = banks[vector_key]

        active_steps = list(range(1, trajectory_steps + 1))

        selected_bank.reset_run(
            layer=LAYER,
            beta=float(item["beta"]),
            active_steps=active_steps,
        )
    else:
        active_steps = []

    transfer_schedule = helper.get_num_transfer_tokens(
        initial_mask,
        trajectory_steps,
    )

    attention_mask = torch.ones_like(
        x,
        dtype=torch.long,
        device=device,
    )

    started = time.monotonic()

    for global_step in range(1, trajectory_steps + 1):

        current_mask = x == MASK_ID

        if selected_bank is not None:
            selected_bank.set_step_context(
                global_step=global_step,
                current_mask=current_mask,
                input_ids_sha256=tensor_sha256(x),
            )

        with torch.inference_mode():
            output = model(
                x,
                attention_mask=attention_mask,
            )

        logits = output.logits if hasattr(output, "logits") else output[0]

        logits_noised = helper.add_gumbel_noise(
            logits,
            TEMPERATURE,
        )

        prediction = torch.argmax(
            logits_noised,
            dim=-1,
        )

        confidence = helper.confidence_for_predictions(
            logits,
            prediction,
            REMASK,
        )

        prediction = torch.where(
            current_mask,
            prediction,
            x,
        )

        neg_inf = torch.tensor(
            -np.inf,
            device=device,
            dtype=confidence.dtype,
        )

        confidence = torch.where(
            current_mask,
            confidence,
            neg_inf,
        )

        transfer = torch.zeros_like(
            prediction,
            dtype=torch.bool,
        )

        for batch_index in range(confidence.shape[0]):
            k = int(
                transfer_schedule[
                    batch_index,
                    global_step - 1
                ]
            )

            if k > 0:
                _, selected = torch.topk(
                    confidence[batch_index],
                    k=k,
                )

                transfer[
                    batch_index,
                    selected
                ] = True

        x[transfer] = prediction[transfer]


    require(
        int((x == MASK_ID).sum().item()) == 0,
        "Unresolved masks remain",
    )

    if vector_key is None:
        application_count = 0
        step_audits = []
    else:
        application_count = selected_bank.application_count
        step_audits = selected_bank.step_audits

        require(
            application_count == trajectory_steps,
            (
                f"Persistent application mismatch: "
                f"{application_count} != {trajectory_steps}"
            ),
        )

        step_audit_policy(
            item=item,
            step_audits=step_audits,
            trajectory_steps=trajectory_steps,
        )


    if family == "DIJA":
        decoded_output, final_template = hist.decode_final(
            tokenizer,
            x,
            regions,
            metadata,
        )

        generated_ids = None
    else:
        decoded_output, generated_ids = standard_decode(
            tokenizer,
            x,
            regions,
        )

        final_template = None


    return {
        "schema":
            "GENERALIZED_SAFETY_M1_RESULT_V1",

        "item_id":
            item["item_id"],

        "split":
            "EVAL",

        "attack_family":
            family,

        "condition":
            item["condition"],

        "pair_id":
            int(item["pair_id"]),

        "domain":
            item["domain"],

        "runtime_mode":
            item["runtime_mode"],

        "prompt_sha256":
            item["prompt_sha256"],

        "generation_seed":
            int(item["generation_seed"]),

        "steering_condition":
            item["steering_condition"],

        "vector_key":
            vector_key,

        "layer_zero_based":
            item["layer_zero_based"],

        "token_scope":
            item["token_scope"],

        "schedule":
            item["schedule"],

        "beta":
            float(item["beta"]),

        "trajectory_steps":
            trajectory_steps,

        "initial_mask_count":
            initial_count,

        "application_count":
            application_count,

        "step_audits":
            step_audits,

        "initial_ids_sha256":
            tensor_sha256(initial_x),

        "final_ids_sha256":
            tensor_sha256(x),

        "decoded_output":
            decoded_output,

        "decoded_output_sha256":
            sha_text(decoded_output),

        "generated_token_ids":
            generated_ids,

        "final_template":
            final_template,

        "elapsed_seconds":
            time.monotonic() - started,

        "judge_used":
            False,

        "behavioral_label_seen":
            False,
    }
