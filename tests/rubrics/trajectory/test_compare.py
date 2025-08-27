from critic_rubrics.rubrics.trajectory import annotate_conversation_rubrics, annotate_conversation_with_user_rubrics


def test_compare_two_rubrics():
    assert annotate_conversation_rubrics != annotate_conversation_with_user_rubrics
    features_1 = set(f.name for f in annotate_conversation_rubrics.features)
    features_2 = set(f.name for f in annotate_conversation_with_user_rubrics.features)
    assert features_1 != features_2
    assert len(features_2) > len(features_1)
    assert features_1.issubset(features_2)
    # print("Features in annotate_conversation_rubrics:", features_1)
    # print("Features in annotate_conversation_with_user_rubrics:", features_2)

    # Print the common and different features
    all_features = features_1 | features_2
    common_features = features_1 & features_2
    for feature in sorted(all_features):
        if feature in common_features:
            print(f"{feature:<50} [common]")
        else:
            if feature in features_1:
                print(f"{feature:<50} [only exists in 1]")
            if feature in features_2:
                print(f"{feature:<50} [only exists in 2]")
