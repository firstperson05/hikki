#!/usr/bin/env python3
"""
FastMind LM - Debug Dataset Generator
Creates 100,000 tokens of structured synthetic text for testing
"""

import random
import sys

def generate_counting_patterns(tokens_needed):
    """Generate counting patterns: 1, 2, 3..."""
    patterns = []
    for i in range(1, 1000):
        patterns.append(f"The number {i} comes after {i-1}. ")
        patterns.append(f"{i} plus {i} equals {i*2}. ")
    return ''.join(patterns)

def generate_alphabet_patterns(tokens_needed):
    """Generate alphabet repetitions"""
    patterns = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for _ in range(100):
        for letter in alphabet:
            patterns.append(f"The letter {letter.upper()} is {letter}. ")
    return ''.join(patterns)

def generate_simple_sentences(tokens_needed):
    """Generate simple structured sentences"""
    subjects = ["The cat", "A dog", "The bird", "A fish", "The child", "A student"]
    verbs = ["runs", "jumps", "sleeps", "eats", "plays", "reads"]
    objects = ["fast", "high", "well", "food", "games", "books"]
    
    patterns = []
    for _ in range(500):
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        patterns.append(f"{subject} {verb} {obj}. ")
    return ''.join(patterns)

def generate_tech_patterns(tokens_needed):
    """Generate technology-related patterns"""
    tech_words = ["computer", "algorithm", "data", "network", "software", "hardware"]
    actions = ["processes", "analyzes", "stores", "transmits", "computes", "optimizes"]
    
    patterns = []
    for _ in range(300):
        tech = random.choice(tech_words)
        action = random.choice(actions)
        patterns.append(f"The {tech} {action} information. ")
        patterns.append(f"{tech.capitalize()} is important for technology. ")
    return ''.join(patterns)

def generate_stories(tokens_needed):
    """Generate coherent short stories"""
    story_templates = [
        "Once upon a time, there was a {character}. The {character} lived in a {place}. "
        "Every day, the {character} would {activity}. One {day_time}, something {event} happened. "
        "The {character} felt {emotion}. After that, the {character} learned to {lesson}. "
        "And they lived {ending} ever after.",
        
        "In a {setting}, a {character} discovered {object}. It was {description}. "
        "The {character} decided to {action}. This led to {consequence}. "
        "Finally, the {character} achieved {resolution}.",
        
        "The {character} had a {problem}. To solve it, they needed {solution}. "
        "First, they tried {attempt1}. When that failed, they tried {attempt2}. "
        "In the end, the {character} succeeded by {success_method}."
    ]
    
    characters = ["brave knight", "wise wizard", "clever princess", "young prince", "old merchant"]
    places = ["small village", "big castle", "dark forest", "magic tower", "busy city"]
    activities = ["explore the woods", "read ancient books", "practice magic", "help others", "solve puzzles"]
    day_time = ["morning", "afternoon", "evening", "night"]
    events = ["magical", "surprising", "wonderful", "challenging", "mysterious"]
    emotions = ["happy", "excited", "curious", "brave", "determined"]
    lessons = ["be kind", "work hard", "never give up", "help others", "stay curious"]
    endings = ["happily", "peacefully", "successfully", "joyfully", "contentedly"]
    
    settings = ["distant kingdom", "modern city", "ancient library", "mysterious island", "space station"]
    objects = ["magic sword", "ancient book", "crystal ball", "flying carpet", "time machine"]
    descriptions = ["shiny and new", "old and dusty", "magical and powerful", "strange and mysterious", "beautiful and elegant"]
    actions = ["investigate further", "use it wisely", "share it with others", "keep it safe", "study its secrets"]
    consequences = ["great discoveries", "amazing adventures", "important lessons", "new friendships", "personal growth"]
    resolutions = ["their goals", "great success", "inner peace", "new knowledge", "true happiness"]
    
    problems = ["difficult challenge", "complex puzzle", "dangerous journey", "important mission", "personal quest"]
    solutions = ["courage", "wisdom", "friends", "determination", "creativity"]
    attempt1 = ["asking for help", "working alone", "using brute force", "trying magic", "following instincts"]
    attempt2 = ["thinking differently", "collaborating", "studying more", "practicing", "seeking guidance"]
    success_method = ["combination of approaches", "unexpected insight", "persistent effort", "creative solution", "teamwork"]
    
    patterns = []
    for _ in range(200):
        template = random.choice(story_templates)
        if template.startswith("Once upon a time"):
            story = template.format(
                character=random.choice(characters),
                place=random.choice(places),
                activity=random.choice(activities),
                day_time=random.choice(day_time),
                event=random.choice(events),
                emotion=random.choice(emotions),
                lesson=random.choice(lessons),
                ending=random.choice(endings)
            )
        elif template.startswith("In a"):
            story = template.format(
                setting=random.choice(settings),
                character=random.choice(characters),
                object=random.choice(objects),
                description=random.choice(descriptions),
                action=random.choice(actions),
                consequence=random.choice(consequences),
                resolution=random.choice(resolutions)
            )
        else:
            story = template.format(
                character=random.choice(characters),
                problem=random.choice(problems),
                solution=random.choice(solutions),
                attempt1=random.choice(attempt1),
                attempt2=random.choice(attempt2),
                success_method=random.choice(success_method)
            )
        patterns.append(story + " ")
    
    return ''.join(patterns)

def main():
    print("🔧 FastMind LM - Debug Dataset Generator")
    print("=====================================")
    
    # Generate different pattern types
    print("📝 Generating structured text patterns...")
    
    patterns = []
    
    # Add counting patterns
    print("  - Counting patterns...")
    patterns.append(generate_counting_patterns(25000))
    
    # Add alphabet patterns  
    print("  - Alphabet patterns...")
    patterns.append(generate_alphabet_patterns(25000))
    
    # Add simple sentences
    print("  - Simple sentences...")
    patterns.append(generate_simple_sentences(25000))
    
    # Add tech patterns
    print("  - Technology patterns...")
    patterns.append(generate_tech_patterns(25000))
    
    # Add coherent stories (NEW!)
    print("  - Coherent stories...")
    patterns.append(generate_stories(25000))
    
    # Combine WITHOUT shuffling to maintain structure
    full_text = ''.join(patterns)
    
    # Write to file
    output_file = "data/debug.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    # Statistics
    token_count = len(full_text.split())
    file_size = len(full_text.encode('utf-8'))
    
    print(f"\n✅ Debug dataset created!")
    print(f"   File: {output_file}")
    print(f"   Tokens: {token_count:,}")
    print(f"   Size: {file_size:,} bytes")
    print(f"   Patterns: counting, alphabet, sentences, technology, stories")
    print(f"\n🎯 Use with configs/tiny.toml for quick testing")
    print(f"   Now with COHERENT text structure!")

if __name__ == "__main__":
    main()
