RESPONSE_TEMPLATES = {
    "Praise": [
        "Thank you so much for your kind words! We're thrilled that you enjoyed it. Your support means the world to us! 🙏",
        "We're so happy to hear that you loved it! Thank you for taking the time to share your feedback. It truly motivates us! 💙",
        "Thank you! Your appreciation fuels our creativity. We're grateful for your support! ✨"
    ],
    
    "Support": [
        "Thank you for your encouragement! Your support keeps us going. We'll keep creating and improving! 💪",
        "Your words mean so much to us! We're committed to bringing you more quality content. Stay tuned! 🌟",
        "Thank you for believing in us! Your support is our driving force. We appreciate you! ❤️"
    ],
    
    "Constructive Criticism": [
        "Thank you for your thoughtful feedback! We really appreciate you taking the time to share your perspective. We'll definitely take your suggestions into consideration for future content. Your input helps us grow! 📝",
        "We value your constructive feedback! It's comments like yours that help us improve. We'll work on incorporating your suggestions. Thank you for being part of our journey! 🎯",
        "Thank you for the detailed feedback! We're always looking to improve, and your insights are incredibly helpful. We'll keep your suggestions in mind for our next projects! 💡"
    ],
    
    "Hate/Abuse": [
        "We're sorry to hear that our content didn't resonate with you. We're always open to constructive feedback if you'd like to share specific concerns. We aim to create content that everyone can enjoy. 🙏",
        "We understand that not every piece of content will appeal to everyone. If you have specific feedback about what didn't work for you, we'd be happy to listen. Thank you for your time. 💙",
        "We're sorry our content didn't meet your expectations. We're committed to continuous improvement and value all forms of respectful feedback. 🌟"
    ],
    
    "Threat": [
        "We take all concerns seriously. If you believe our content violates any platform guidelines, please report it through the official channels. We're committed to following all community standards. Thank you for bringing this to our attention. 📋",
        "We understand your concern. If you have specific issues with our content, please reach out through our official channels, and we'll be happy to address them. We're committed to maintaining a respectful community. 🤝",
        "Thank you for your feedback. We take platform guidelines seriously and are committed to compliance. If you have specific concerns, please contact us directly, and we'll address them promptly. 📧"
    ],
    
    "Emotional": [
        "We're deeply touched that our content resonated with you on such a personal level. Thank you for sharing that with us. It's moments like these that remind us why we create. 💙",
        "Your words moved us. We're honored that our content could connect with you in such a meaningful way. Thank you for being part of our community. ❤️",
        "We're so glad this touched your heart. Creating content that connects emotionally is one of our greatest goals. Thank you for sharing your experience with us! ✨"
    ],
    
    "Irrelevant/Spam": [
        "Thank you for your comment. We focus on maintaining a space for meaningful discussions about our content. If you have feedback about our work, we'd love to hear it! 🎯"
    ],
    
    "Question/Suggestion": [
        "Great question! We appreciate your interest. We'll definitely consider your suggestion for future content. Thank you for the idea! 💡",
        "Thank you for your suggestion! We love hearing ideas from our community. We'll keep this in mind for upcoming projects. Stay tuned! 🌟",
        "That's a wonderful idea! We're always looking for new directions to explore. Thank you for sharing your thoughts with us! We'll definitely consider it! ✨"
    ]
}


def get_response_template(category, template_index=0):
    if category not in RESPONSE_TEMPLATES:
        return "Thank you for your comment! We appreciate your feedback."
    
    templates = RESPONSE_TEMPLATES[category]
    index = template_index % len(templates)
    return templates[index]


def generate_response(comment, category, use_template=True):
    if use_template:
        import random
        return get_response_template(category, random.randint(0, len(RESPONSE_TEMPLATES.get(category, [])) - 1))
    else:
        return get_response_template(category)


if __name__ == "__main__":
    print("Response Template Examples:")
    print("=" * 60)
    
    for category, templates in RESPONSE_TEMPLATES.items():
        print(f"\n{category}:")
        print(f"  {templates[0]}")
