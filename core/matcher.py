import ahocorasick

# class KeywordMatcher:
#     def __init__(self, keywords):
#         self.A = ahocorasick.Automaton()
#         for idx, word in enumerate(keywords):
#             self.A.add_word(word, (idx, word))
#         self.A.make_automaton()
#         print(f"已加载 {len(keywords)} 个关键词，构建完成。")

#     def find_any(self, text):
#         for item in self.A.iter(text):
#             found_word = item[1][1]
#             return True, found_word 
#         return False, None

#     def find_all(self, text):
#         results = []
#         for item in self.A.iter(text):
#             results.append(item[1][1])
#         return results

import ahocorasick

class KeywordMatcher:
    def __init__(self, mapping_dict):
        self.A = ahocorasick.Automaton()
        count = 0
        
        if isinstance(mapping_dict, dict):
            for parent, children in mapping_dict.items():
                self.A.add_word(parent, (parent, parent))
                count += 1
                
                for child in children:
                    self.A.add_word(child, (child, parent))
                    count += 1
        else:
            for idx, word in enumerate(mapping_dict):
                self.A.add_word(word, (idx, word))
                count += 1
                
        self.A.make_automaton()
        print(f"已加载 {count} 个检索词映射，AC自动机构建完成。")

    def find_any(self, text):
        for item in self.A.iter(text):
            found_parent_word = item[1][1]
            return True, found_parent_word 
        return False, None

    def find_all(self, text):
        results = []
        for item in self.A.iter(text):
            parent_word = item[1][1]
            results.append(parent_word)
            
        return list(set(results))