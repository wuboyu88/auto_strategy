import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class GreedyStrategy:
    def __init__(self, df_train, target, var_dict, expected_pass_rate, initial_pass_rate=0.98, cached_y_pred=None,
                 df_test=None, show=True, save_path=None, logging=True):
        """
        初始化
        :param df_train: 训练集
        :param target: 目标变量
        :param var_dict: 变量方向字典
        :param expected_pass_rate: 目标通过率
        :param initial_pass_rate: 初始通过率
        :param cached_y_pred: 前置规则预测结果
        :param df_test: 测试集
        :param show: 是否画图
        :param save_path: 图片保存路径
        :param logging: 是否打印日志
        """
        self.df_train = df_train
        self.target = target
        self.var_dict = var_dict
        self.expected_pass_rate = expected_pass_rate
        self.initial_pass_rate = initial_pass_rate
        self.cached_y_pred = cached_y_pred
        self.df_test = df_test
        self.show = show
        self.save_path = save_path
        self.logging = logging
        self.cut_off_dict = None

        # 1.如果单条规则在目标通过率的情况下没有坏样本则不考虑该变量
        # 2.如果单条规则通过率达不到目标通过率也不考虑该变量（比如变量只有1个值）
        ordered_rule = self.get_ordered_var_by_pass_rate(current_pass_rate=self.expected_pass_rate,
                                                         cached_y_pred=self.cached_y_pred)
        self.var_dict = OrderedDict()
        for var, [_, recall] in ordered_rule:
            if recall != 0:
                self.var_dict[var] = var_dict[var]

    @staticmethod
    def combine_two_list_by_or(list1, list2):
        """
        对两个list取逻辑或操作
        :param list1: [0, 1, 0, 1]
        :param list2: [0, 0, 1, 1]
        :return: [0, 1, 1, 1]
        """
        return [ele[0] or ele[1] for ele in zip(list1, list2)]

    @staticmethod
    def get_pass_rate_and_recall(y_true, y_pred):
        """
        计算通过率和召回率
        :param y_true: [1, 0, 0, 1]
        :param y_pred: [1, 0, 0, 0]
        :return: 0.75, 0.5
        """
        # 通过率和召回率
        confusion_mat = confusion_matrix(y_true, y_pred, labels=[1, 0])
        tp = confusion_mat[0][0]
        fp = confusion_mat[1][0]
        fn = confusion_mat[0][1]
        tn = confusion_mat[1][1]

        pass_rate = (fn + tn) / (tp + fp + fn + tn)
        recall = tp / (tp + fn)
        return pass_rate, recall

    def get_ordered_var_by_pass_rate(self, current_pass_rate, used_var_dict=None, cached_y_pred=None):
        """
        指定通过率和前置规则预测结果的情况下，按照召回率从高到低给变量排序
        :param current_pass_rate: 当前通过率
        :param used_var_dict：当前可用变量字典
        :param cached_y_pred: 前置规则预测结果
        :return: (var, [search_domain, recall])
        """
        result = dict()
        y_true = self.df_train[self.target]
        used_var_dict = self.var_dict if used_var_dict is None else used_var_dict
        for var, sign in used_var_dict.items():
            values = self.df_train[var].tolist()
            if sign == 1:
                tmp = np.percentile(values, current_pass_rate * 100)
                search_domain = sorted(set([ele for ele in values if ele > tmp]))

                while len(search_domain) > 0:
                    lower_bound = search_domain[0]
                    y_pred = np.where(self.df_train[var] > lower_bound, 1, 0).tolist()
                    # y_pred = self.df_train[var].apply(lambda x: 1 if x > lower_bound else 0).tolist()

                    if cached_y_pred is not None:
                        y_pred = self.combine_two_list_by_or(y_pred, cached_y_pred)
                    pass_rate, recall = self.get_pass_rate_and_recall(y_true, y_pred)
                    if pass_rate >= current_pass_rate:
                        result[var] = [search_domain, recall]
                        break
                    else:
                        search_domain.pop(0)

            else:
                tmp = np.percentile(values, (1 - current_pass_rate) * 100)
                search_domain = sorted(set([ele for ele in values if ele < tmp]))

                while len(search_domain) > 0:
                    upper_bound = search_domain[-1]
                    y_pred = np.where(self.df_train[var] < upper_bound, 1, 0).tolist()
                    # y_pred = self.df_train[var].apply(lambda x: 1 if x < upper_bound else 0).tolist()

                    if cached_y_pred is not None:
                        y_pred = self.combine_two_list_by_or(y_pred, cached_y_pred)
                    pass_rate, recall = self.get_pass_rate_and_recall(y_true, y_pred)
                    if pass_rate >= current_pass_rate:
                        result[var] = [search_domain, recall]
                        break
                    else:
                        search_domain.pop(-1)

        result = sorted(result.items(), key=lambda x: x[1][1], reverse=True)
        return result

    def get_greedy_cut_off_dynamic(self):
        """
        贪心算法找出最优阈值(动态)
        1.从第一个变量开始，第一个变量通过率98%，然后依次降低到95%，如果是10个变量，则截止第二个变量的整体通过率为98%-(98%-95%)/10
        2.第k+1次迭代: 固定前k次的规则组合预测结果，优化目标是通过加入第k+1个变量，降低当前通过率，使得整体召回率最大
        :return: {var: [cut_off, recall, pass_rate]}
        """
        y_true = self.df_train[self.target]
        cached_y_pred = self.cached_y_pred
        greedy_cut_off_dict = OrderedDict()
        used_var_dict = deepcopy(self.var_dict)

        i = 0
        while i < len(self.var_dict):
            current_pass_rate = self.initial_pass_rate - (self.initial_pass_rate - self.expected_pass_rate) / len(
                self.var_dict) * i
            ordered_var = self.get_ordered_var_by_pass_rate(current_pass_rate, used_var_dict, cached_y_pred)
            if len(ordered_var) == 0:
                break
            var, [search_domain, _] = ordered_var[0]

            tmp = []

            for cut_off in search_domain:
                if self.var_dict[var] == 1:
                    y_pred = np.where(self.df_train[var] > cut_off, 1, 0).tolist()
                    # y_pred = self.df_train[var].apply(lambda x: 1 if x > cut_off else 0).tolist()
                else:
                    y_pred = np.where(self.df_train[var] < cut_off, 1, 0).tolist()
                    # y_pred = self.df_train[var].apply(lambda x: 1 if x < cut_off else 0).tolist()
                if cached_y_pred is not None:
                    y_pred = self.combine_two_list_by_or(y_pred, cached_y_pred)
                pass_rate, recall = self.get_pass_rate_and_recall(y_true, y_pred)
                if pass_rate >= current_pass_rate:
                    tmp.append((recall, pass_rate, cut_off, y_pred))

            if len(tmp) == 0:
                print(
                    "{}'s search_domain fails to achieve the expected pass rate, needs to change".format(var))
                return
            else:
                used_var_dict.pop(var)
                recall, pass_rate, cut_off, y_pred = sorted(tmp, key=lambda x: (x[0], x[1]), reverse=True)[0]
                cached_y_pred = y_pred
                greedy_cut_off_dict[var] = cut_off

            i += 1
        return greedy_cut_off_dict

    def get_rule_pass_rate_and_recall(self, cut_off_dict, df=None, is_cum=False):
        """
        根据规则的切分点,得到单条规则的通过率和召回率
        :param cut_off_dict: {var: cuf_off, ...}
        :param df: 数据集
        :param is_cum: 累计规则的预测结果
        :return: {var: [pass_rate, recall]}
        """
        result = OrderedDict()
        df = self.df_train if df is None else df
        y_true = df[self.target]
        cached_y_pred = self.cached_y_pred
        for var, cut_off in cut_off_dict.items():
            if self.var_dict[var] == 1:
                y_pred = np.where(df[var] > cut_off, 1, 0).tolist()
                # y_pred = df[var].apply(lambda x: 1 if x > cut_off else 0).tolist()
            else:
                y_pred = np.where(df[var] < cut_off, 1, 0).tolist()
                # y_pred = df[var].apply(lambda x: 1 if x < cut_off else 0).tolist()
            if is_cum and cached_y_pred:
                y_pred = self.combine_two_list_by_or(y_pred, cached_y_pred)
            cached_y_pred = y_pred
            result[var] = self.get_pass_rate_and_recall(y_true, y_pred)
        return result

    def get_all_rule_pass_rate_and_recall(self, cut_off_dict, df=None):
        """
        根据规则的切分点,得到整个规则集的通过率和召回率
        :param cut_off_dict: {var: cuf_off, ...}
        :param df: 数据集
        :return: pass_rate, recall
        """
        df = self.df_train if df is None else df
        y_true = df[self.target]
        tmp = None
        for var, cut_off in cut_off_dict.items():
            if self.var_dict[var] == 1:
                y_pred = np.where(df[var] > cut_off, 1, 0).tolist()
                # y_pred = df[var].apply(lambda x: 1 if x > cut_off else 0).tolist()
            else:
                y_pred = np.where(df[var] < cut_off, 1, 0).tolist()
                # y_pred = df[var].apply(lambda x: 1 if x < cut_off else 0).tolist()
            if tmp is None:
                tmp = y_pred
            else:
                tmp = self.combine_two_list_by_or(y_pred, tmp)
        return self.get_pass_rate_and_recall(y_true, tmp)

    def plot_result(self, rule_pass_rate_and_recall_dict):
        """
        画出规则集的通过率和召回率
        :param rule_pass_rate_and_recall_dict: 规则集结果 {var: [pass_rate, recall]}
        :return:
        """
        pass_rate_list = []
        recall_list = []
        for ele in rule_pass_rate_and_recall_dict.values():
            pass_rate_list.append(ele[0])
            recall_list.append(ele[1])
        step_list = list(range(len(pass_rate_list)))
        plt.clf()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.xlabel('贪心算法迭代次数')
        plt.ylabel('百分比')
        plt.plot(step_list, pass_rate_list, label='通过率')
        plt.plot(step_list, recall_list, label='召回率')
        plt.axhline(y=self.expected_pass_rate, ls="--", c='black', label='目标通过率{}'.format(self.expected_pass_rate))
        plt.legend(loc='best')
        plt.grid(ls=":", c='b')
        plt.title('贪心算法')
        if self.save_path is not None:
            plt.savefig(self.save_path)

        if self.show:
            plt.show()

        plt.close()

    def fine_tune(self):
        """
        在执行完贪心算法之后进行微调
        适当降低通过率（不低于目标通过率的情况下）来提高召回率：原因是经过贪心算法之后，最终通过率可能高于目标通过率，还有下探空间
        :return:
        """
        greedy_cut_off_dict = self.get_greedy_cut_off_dynamic()
        used_var_dict = {}
        for var in self.get_greedy_cut_off_dynamic().keys():
            used_var_dict[var] = self.var_dict[var]

        ordered_var = self.get_ordered_var_by_pass_rate(self.expected_pass_rate, used_var_dict, self.cached_y_pred)

        global_pass_rate, global_recall = self.get_all_rule_pass_rate_and_recall(greedy_cut_off_dict, self.df_train)

        # 2.循环操作
        cut_off_dict = deepcopy(greedy_cut_off_dict)
        for ele in ordered_var:
            var, [search_domain, _] = ele
            init_cut_off = greedy_cut_off_dict[var]
            is_tuned = False
            if self.var_dict[var] == 1:
                search_domain = [x for x in search_domain if x < init_cut_off]
            else:
                search_domain = [x for x in search_domain if x > init_cut_off]

            for cut_off in search_domain:
                cut_off_dict[var] = cut_off
                pass_rate, recall = self.get_all_rule_pass_rate_and_recall(cut_off_dict, self.df_train)
                if pass_rate >= self.expected_pass_rate and recall > global_recall:
                    if self.logging:
                        print('{}调整前: cut_off={}, pass_rate={}, recall={}'.format(var, init_cut_off, global_pass_rate,
                                                                                  global_recall))
                        print('{}调整后: cut_off={}, pass_rate={}, recall={}'.format(var, cut_off, pass_rate, recall))
                    is_tuned = True
                    global_pass_rate, global_recall = pass_rate, recall
                    break
            if not is_tuned:
                cut_off_dict[var] = init_cut_off
        return cut_off_dict

    def get_rule_detail(self, rule_pass_rate_and_recall, rule_pass_rate_and_recall_cum, cut_off_dict=None):
        """
        得到规则明细
        :param rule_pass_rate_and_recall: 单条规则
        :param rule_pass_rate_and_recall_cum: 累计情况
        :param cut_off_dict: 阈值
        :return:
        """
        cut_off_dict = self.cut_off_dict if cut_off_dict is None else cut_off_dict
        rule_pass_rate_and_recall = pd.DataFrame(rule_pass_rate_and_recall).T
        rule_pass_rate_and_recall_cum = pd.DataFrame(rule_pass_rate_and_recall_cum).T
        rule_detail = pd.concat([rule_pass_rate_and_recall, rule_pass_rate_and_recall_cum], axis=1)
        rule_detail.columns = ['单条规则通过率', '单条规则召回率', '累计通过率', '累计召回率']
        for var, cut_off in cut_off_dict.items():
            if self.var_dict[var] == 1:
                rule_name = '{} > {}'.format(var, cut_off)
            else:
                rule_name = '{} < {}'.format(var, cut_off)
            rule_detail.loc[var, '规则名称'] = rule_name
        rule_detail = rule_detail.reset_index()
        rule_detail.columns = ['变量名称', '单条规则通过率', '单条规则召回率', '累计通过率', '累计召回率', '规则名称']
        rule_detail = rule_detail[['变量名称', '规则名称', '单条规则通过率', '单条规则召回率', '累计通过率', '累计召回率']]
        return rule_detail

    def fit(self):
        """
        通过训练集找出最优阈值，并计算整体通过情况和规则明细
        :return:
        """
        cut_off_dict = self.fine_tune()
        self.cut_off_dict = cut_off_dict

        rule_pass_rate_and_recall = self.get_rule_pass_rate_and_recall(cut_off_dict)
        rule_pass_rate_and_recall_cum = self.get_rule_pass_rate_and_recall(cut_off_dict, is_cum=True)
        self.plot_result(rule_pass_rate_and_recall_cum)

        rule_detail = self.get_rule_detail(rule_pass_rate_and_recall, rule_pass_rate_and_recall_cum, cut_off_dict)
        all_rule_pass_rate_and_recall = self.get_all_rule_pass_rate_and_recall(cut_off_dict)

        return all_rule_pass_rate_and_recall, rule_detail

    def predict(self, df=None, cut_off_dict=None):
        """
        将训练集找出的最优规则集应用与测试集，并计算测试集整体通过情况和规则明细
        :param df: 数据集
        :param cut_off_dict: 阈值
        :return:
        """
        if df is None:
            df = self.df_test or self.df_train
        cut_off_dict = self.cut_off_dict if cut_off_dict is None else cut_off_dict

        rule_pass_rate_and_recall = self.get_rule_pass_rate_and_recall(cut_off_dict, df)
        rule_pass_rate_and_recall_cum = self.get_rule_pass_rate_and_recall(cut_off_dict, df, is_cum=True)
        self.plot_result(rule_pass_rate_and_recall_cum)

        rule_detail = self.get_rule_detail(rule_pass_rate_and_recall, rule_pass_rate_and_recall_cum, cut_off_dict)
        all_rule_pass_rate_and_recall = self.get_all_rule_pass_rate_and_recall(cut_off_dict, df)

        return all_rule_pass_rate_and_recall, rule_detail
