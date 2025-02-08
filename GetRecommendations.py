def get_recommendation(score):
    if score > 70:
        return "Content likely violates Meta Community Standards. Major modifications needed."
    elif score > 30:
        return "Content may need modifications to comply with Meta Community Standards."
    else:
        return "Content likely complies with Meta Community Standards."