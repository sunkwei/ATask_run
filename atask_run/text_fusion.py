
class TextFusion:
    """基于最大公共子串的文本融合器"""
    
    def __init__(self, mismatch_threshold=2):
        self.mismatch_threshold = mismatch_threshold  # 连续不匹配阈值
    
    def find_fusion_boundaries(self, text1, text2, common_sub):
        """基于连续不匹配的边界检测"""
        # 找到公共子串位置
        pos1 = text1.find(common_sub)
        pos2 = text2.find(common_sub)
        
        if pos1 == -1 or pos2 == -1:
            return 0, len(text1)  # 如果没有公共子串，返回整个文本
        
        if pos1 == 0 or pos2 == 0:
            # 公共子串在开头，前段为空
            front_boundary = pos1
        else:
            front_boundary = self._find_front_boundary(text1, text2, pos1, pos2)
        
        # 检查公共子串是否在末尾
        if pos1 + len(common_sub) >= len(text1) or pos2 + len(common_sub) >= len(text2):
            # 公共子串在末尾，后段为空
            back_boundary = pos1 + len(common_sub) + 1
        else:
            back_boundary = self._find_back_boundary(text1, text2, pos1, pos2, len(common_sub))
        
        return front_boundary, back_boundary
    
    def _find_front_boundary(self, text1, text2, pos1, pos2):
        """向前寻找融合边界"""
        i, j = pos1 - 1, pos2 - 1
        consecutive_mismatch = 0
        
        while i >= 0 and j >= 0:
            if text1[i] == text2[j]:
                consecutive_mismatch = 0  # 重置计数器
            else:
                consecutive_mismatch += 1
                if consecutive_mismatch >= self.mismatch_threshold:
                    if i + self.mismatch_threshold >= len(text1):
                        return len(text1)  # 到达文本末尾
                    return i + self.mismatch_threshold
            i -= 1
            j -= 1

        if consecutive_mismatch < self.mismatch_threshold:

            return i + consecutive_mismatch + 1
        return 0  # 一直匹配到开头
    
    def _find_back_boundary(self, text1, text2, pos1, pos2, common_len):
        """向后寻找融合边界"""
        i, j = pos1 + common_len, pos2 + common_len
        consecutive_mismatch = 0
        text1_len, text2_len = len(text1), len(text2)
        
        while i < text1_len and j < text2_len:
            
            if text1[i] == text2[j]:
                consecutive_mismatch = 0  # 重置计数器
            else:
                consecutive_mismatch += 1
                if consecutive_mismatch >= self.mismatch_threshold:
                    # if i - consecutive_mismatch + 1 < 0:
                    #     return 0  # 到达文本开头
                    return i - consecutive_mismatch + 1
            i += 1
            j += 1

        if consecutive_mismatch < self.mismatch_threshold:
            return i - consecutive_mismatch
        return len(text1)  # 一直匹配到结尾
    
    def smart_fusion(self, text1, text2, common_sub):
        """智能文本融合"""
        
        if common_sub not in text1 or common_sub not in text2:
            return ''
        
        # 2. 找到融合边界
        front_boundary, back_boundary = self.find_fusion_boundaries(text1, text2, common_sub)

        # 提取各段内容（增强边界检查）
        common_start = text1.find(common_sub)
        # 确保边界有效
        front_boundary = max(0, min(front_boundary, common_start))
        back_boundary = max(common_start + len(common_sub), 
                           min(back_boundary, len(text1)))
        
        front_part = text1[front_boundary:common_start]
        back_part = text1[common_start + len(common_sub):back_boundary]
        
        # 组合最终结果
        result = front_part + common_sub + back_part
        
        return result
    
    def batch_fusion(self, text_pairs):
        """批量文本融合"""
        results = []
        for text1, text2 in text_pairs:
            result = self.smart_fusion(text1, text2)
            results.append(result)
        return results
    
    def __call__(self, text1, text2, common_sub_list):
        
        R = ""
        for common_sub in common_sub_list:
            result = self.smart_fusion(text1, text2, common_sub)
            
            if len(result) > len(R):
                R = result
        
        return R

if __name__ == "__main__":
    # 运行测试
    fusion = TextFusion()
    text1 = "44大家也要学好待6666"
    text2 = "946大家也要学好待回"
    common_sub = ["大家也要学好待"]
    print(fusion(text1, text2, common_sub))

