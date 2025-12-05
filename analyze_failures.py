# Analyze which byName tests are failing
# Failing tests: 5, 6, 9, 10, 11, 12, 14, 15, 17, 19, 20, 21, 22, 27
# Passing tests: 0, 1, 2, 3, 4, 7, 8, 13, 16, 18, 23, 24, 25, 26, 28, 29

failing_indices = [5, 6, 9, 10, 11, 12, 14, 15, 17, 19, 20, 21, 22, 27]
passing_indices = [0, 1, 2, 3, 4, 7, 8, 13, 16, 18, 23, 24, 25, 26, 28, 29]

# These are the test names from earlier logs
test_names = [
    "google-research-bert",  # 0 - PASSING
    "bookcorpus",  # 1 - PASSING
    "bert-base-uncased",  # 2 - PASSING
    "audience_classifier_model",  # 3 - PASSING
    "openai-whisper",  # 4 - PASSING
    "transformers-research-projects-distillation",  # 5 - FAILING
    "rajpurkar-squad",  # 6 - FAILING
    "distilbert-base-uncased-distilled-squad",  # 7 - PASSING
    "mv-lab-swin2sr",  # 8 - PASSING
    "hliang001-flickr2k",  # 9 - FAILING
    "caidas-swin2SR-lightweight-x2-64",  # 10 - FAILING
    "moondream",  # 11 - FAILING
    "vikhyatk-moondream2",  # 12 - FAILING
    "microsoft-git",  # 13 - PASSING
    "microsoft-git-base",  # 14 - FAILING
    "WinKawaks-vit-tiny-patch16-224",  # 15 - FAILING
    "fashion-clip",  # 16 - PASSING
    "fashion-mnist",  # 17 - FAILING
    "patrickjohncyh-fashion-clip",  # 18 - PASSING
    "lerobot",  # 19 - FAILING
    "lerobot-pusht",  # 20 - FAILING
    "lerobot-diffusion_pusht",  # 21 - FAILING
    "ptm-recommendation-with-transformers",  # 22 - FAILING
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",  # 23 - PASSING
    "KaimingHe-deep-residual-networks",  # 24 - PASSING
    "imagenet-1k",  # 25 - PASSING
    "resnet-50",  # 26 - PASSING
    "fairface",  # 27 - FAILING
    "trained-gender",  # 28 - PASSING
    "trained-gender-ONNX",  # 29 - PASSING
]

print("FAILING byName tests:")
for idx in failing_indices:
    print(f"  Test {idx}: {test_names[idx]}")

print("\nPASSING byName tests:")
for idx in passing_indices:
    print(f"  Test {idx}: {test_names[idx]}")

# Let's look for patterns
print("\n" + "=" * 60)
print("PATTERN ANALYSIS:")
print("=" * 60)

# Check if hyphen count matters
failing_hyphens = [test_names[i].count('-') for i in failing_indices]
passing_hyphens = [test_names[i].count('-') for i in passing_indices]

print(f"\nFailing tests hyphen counts: {failing_hyphens}")
print(f"Passing tests hyphen counts: {passing_hyphens}")

# Check for specific patterns
print("\nChecking if name has:")
for idx in failing_indices:
    name = test_names[idx]
    print(f"  FAIL {idx}: '{name}' - hyphens:{name.count('-')}, underscores:{name.count('_')}, digits:{sum(c.isdigit() for c in name)}")

print("\n")
for idx in passing_indices[:10]:  # Show first 10 passing
    name = test_names[idx]
    print(f"  PASS {idx}: '{name}' - hyphens:{name.count('-')}, underscores:{name.count('_')}, digits:{sum(c.isdigit() for c in name)}")
