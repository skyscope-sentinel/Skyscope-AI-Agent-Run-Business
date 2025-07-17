import AiAssistantAppLogo from '@renderer/assets/images/apps/360-ai.png'
import AiSearchAppLogo from '@renderer/assets/images/apps/ai-search.png'
import BaiduAiAppLogo from '@renderer/assets/images/apps/baidu-ai.png'
import BaicuanAppLogo from '@renderer/assets/images/apps/baixiaoying.webp'
import BoltAppLogo from '@renderer/assets/images/apps/bolt.svg'
import DevvAppLogo from '@renderer/assets/images/apps/devv.png'
import DoubaoAppLogo from '@renderer/assets/images/apps/doubao.png'
import FeloAppLogo from '@renderer/assets/images/apps/felo.png'
import GeminiAppLogo from '@renderer/assets/images/apps/gemini.png'
import HuggingChatLogo from '@renderer/assets/images/apps/huggingchat.svg'
import KimiAppLogo from '@renderer/assets/images/apps/kimi.jpg'
import MetasoAppLogo from '@renderer/assets/images/apps/metaso.webp'
import PerplexityAppLogo from '@renderer/assets/images/apps/perplexity.webp'
import PoeAppLogo from '@renderer/assets/images/apps/poe.webp'
import ZhipuProviderLogo from '@renderer/assets/images/apps/qingyan.png'
import SensetimeAppLogo from '@renderer/assets/images/apps/sensetime.png'
import SparkDeskAppLogo from '@renderer/assets/images/apps/sparkdesk.png'
import TiangongAiLogo from '@renderer/assets/images/apps/tiangong.png'
import WanZhiAppLogo from '@renderer/assets/images/apps/wanzhi.jpg'
import TencentYuanbaoAppLogo from '@renderer/assets/images/apps/yuanbao.png'
import YuewenAppLogo from '@renderer/assets/images/apps/yuewen.png'
import ZhihuAppLogo from '@renderer/assets/images/apps/zhihu.png'
import ClaudeAppLogo from '@renderer/assets/images/models/claude.png'
import HailuoModelLogo from '@renderer/assets/images/models/hailuo.png'
import QwenModelLogo from '@renderer/assets/images/models/qwen.png'
import DeepSeekProviderLogo from '@renderer/assets/images/providers/deepseek.png'
import GroqProviderLogo from '@renderer/assets/images/providers/groq.png'
import OpenAiProviderLogo from '@renderer/assets/images/providers/openai.png'
import SiliconFlowProviderLogo from '@renderer/assets/images/providers/silicon.png'
import MinApp from '@renderer/components/MinApp'
import { MinAppType } from '@renderer/types'

const _apps: MinAppType[] = [
  {
    id: 'openai',
    name: 'ChatGPT',
    url: 'https://chatgpt.com/',
    logo: OpenAiProviderLogo,
    bodered: true
  },
  {
    id: 'gemini',
    name: 'Gemini',
    url: 'https://gemini.google.com/',
    logo: GeminiAppLogo
  },
  {
    id: 'silicon',
    name: 'SiliconFlow',
    url: 'https://cloud.siliconflow.cn/playground/chat',
    logo: SiliconFlowProviderLogo
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    url: 'https://chat.deepseek.com/',
    logo: DeepSeekProviderLogo
  },
  {
    id: 'yi',
    name: '万知',
    url: 'https://www.wanzhi.com/',
    logo: WanZhiAppLogo,
    bodered: true
  },
  {
    id: 'zhipu',
    name: '智谱清言',
    url: 'https://chatglm.cn/main/alltoolsdetail',
    logo: ZhipuProviderLogo
  },
  {
    id: 'moonshot',
    name: 'Kimi',
    url: 'https://kimi.moonshot.cn/',
    logo: KimiAppLogo
  },
  {
    id: 'baichuan',
    name: '百小应',
    url: 'https://ying.baichuan-ai.com/chat',
    logo: BaicuanAppLogo
  },
  {
    id: 'dashscope',
    name: '通义千问',
    url: 'https://tongyi.aliyun.com/qianwen/',
    logo: QwenModelLogo
  },
  {
    id: 'stepfun',
    name: '跃问',
    url: 'https://yuewen.cn/chats/new',
    logo: YuewenAppLogo,
    bodered: true
  },
  {
    id: 'doubao',
    name: '豆包',
    url: 'https://www.doubao.com/chat/',
    logo: DoubaoAppLogo
  },
  {
    id: 'minimax',
    name: '海螺',
    url: 'https://hailuoai.com/',
    logo: HailuoModelLogo
  },
  {
    id: 'groq',
    name: 'Groq',
    url: 'https://chat.groq.com/',
    logo: GroqProviderLogo
  },
  {
    id: 'anthropic',
    name: 'Claude',
    url: 'https://claude.ai/',
    logo: ClaudeAppLogo
  },
  {
    id: '360-ai-so',
    name: '360AI搜索',
    logo: AiSearchAppLogo,
    url: 'https://so.360.com/'
  },
  {
    id: '360-ai-bot',
    name: 'AI 助手',
    logo: AiAssistantAppLogo,
    url: 'https://bot.360.com/',
    bodered: true
  },
  {
    id: 'baidu-ai-chat',
    name: '文心一言',
    logo: BaiduAiAppLogo,
    url: 'https://yiyan.baidu.com/'
  },
  {
    id: 'tencent-yuanbao',
    name: '腾讯元宝',
    logo: TencentYuanbaoAppLogo,
    url: 'https://yuanbao.tencent.com/chat',
    bodered: true
  },
  {
    id: 'sensetime-chat',
    name: '商量',
    logo: SensetimeAppLogo,
    url: 'https://chat.sensetime.com/wb/chat',
    bodered: true
  },
  {
    id: 'spark-desk',
    name: 'SparkDesk',
    logo: SparkDeskAppLogo,
    url: 'https://xinghuo.xfyun.cn/desk'
  },
  {
    id: 'metaso',
    name: '秘塔AI搜索',
    logo: MetasoAppLogo,
    url: 'https://metaso.cn/'
  },
  {
    id: 'poe',
    name: 'Poe',
    logo: PoeAppLogo,
    url: 'https://poe.com'
  },
  {
    id: 'perplexity',
    name: 'perplexity',
    logo: PerplexityAppLogo,
    url: 'https://www.perplexity.ai/'
  },
  {
    id: 'devv',
    name: 'DEVV_',
    logo: DevvAppLogo,
    url: 'https://devv.ai/'
  },
  {
    id: 'tiangong-ai',
    name: '天工AI',
    logo: TiangongAiLogo,
    url: 'https://www.tiangong.cn/',
    bodered: true
  },
  {
    id: 'zhihu-zhiada',
    name: '知乎直答',
    logo: ZhihuAppLogo,
    url: 'https://zhida.zhihu.com/',
    bodered: true
  },
  {
    id: 'hugging-chat',
    name: 'HuggingChat',
    logo: HuggingChatLogo,
    url: 'https://huggingface.co/chat/',
    bodered: true
  },
  {
    id: 'Felo',
    name: 'Felo',
    logo: FeloAppLogo,
    url: 'https://felo.ai/',
    bodered: true
  },
  {
    id: 'bolt',
    name: 'bolt',
    logo: BoltAppLogo,
    url: 'https://bolt.new/',
    bodered: true
  }
]

export function getAllMinApps() {
  return _apps as MinAppType[]
}

export function startMinAppById(id: string) {
  const app = getAllMinApps().find((app) => app?.id === id)
  app && MinApp.start(app)
}
