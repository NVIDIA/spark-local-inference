from qualify.speculative_benchmark import round_metrics


def test_round_metrics_reads_width_16_rounds():
    # A DFlash2 drafter runs speculative rounds at width 16; the shape rows must carry them.
    delta = {'flare_speculative_rounds_total{batch="1",width="16"}': 22,
             'flare_speculative_offered_drafts_total{batch="1",width="16"}': 122,
             'flare_speculative_accepted_drafts_total{batch="1",width="16"}': 77}
    shapes = round_metrics(delta)["shapes"]
    assert [(row["batch"], row["width"]) for row in shapes] == [(1, 16)]
    assert (shapes[0]["offered_drafts"], shapes[0]["accepted_drafts"], shapes[0]["rounds"]) == (122, 77, 22)
