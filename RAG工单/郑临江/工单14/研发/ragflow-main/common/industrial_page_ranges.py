"""工单14：用户页码[1,N]闭区间转换为Python[0,N)半开区间，不丢末页。"""
def normalize_pdf_page_range(start,end,total):
    if total<0:raise ValueError('total不能为负数')
    left=max(0,int(start)-1);right=min(int(end),int(total))
    return min(left,total),max(min(left,total),right)
