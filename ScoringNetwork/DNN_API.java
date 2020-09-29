import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;

import org.json.JSONObject;

import palmhaidian.util.HttpUtil;

public class Main6 {
	public static void main(String[] args)
	{
		Main6 m = new Main6();
		String access_token=m.gettoken();
		System.out.println(access_token);
		//String access_token = "";
		try {
			m.get_text("2233娘是否信仰♂哲学？", "https://aip.baidubce.com/rest/2.0/antispam/v2/spam", access_token);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
 
	public String gettoken() {
		// 官网获取的 API Key 更新为你注册的
		String clientId = "BhVBrZNxRZTGf7SCyGhCiw9c";
		// 官网获取的 Secret Key 更新为你注册的
		String clientSecret = "w2rV6yMU1EUBSIKy2Tfrixbs7dZjrfWH";
		return getAuth(clientId, clientSecret);
	}
	
	public static String getAuth(String ak, String sk) {
		//
		String authHost = "https://aip.baidubce.com/oauth/2.0/token?";
		String getAccessTokenUrl = authHost
				// 1. grant_type为固定参数
				+ "grant_type=client_credentials"
				// 2. 官网获取的 API Key
				+ "&client_id=" + ak
				// 3. 官网获取的 Secret Key
				+ "&client_secret=" + sk;
		try {
			URL realUrl = new URL(getAccessTokenUrl);
			// 打开和URL之间的连接
			HttpURLConnection connection = (HttpURLConnection) realUrl.openConnection();
			connection.setRequestMethod("GET");
			connection.connect();
			// 获取所有响应头字段
			//Map<String, List<String>> map = connection.getHeaderFields();
			// 定义 BufferedReader输入流来读取URL的响应
			BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));
			String result = "";
			String line;
			while ((line = in.readLine()) != null) {
				result += line;
			}
			/**
			 * 返回结果示例
			 */
			JSONObject jsonObject = new JSONObject(result);
			String access_token = jsonObject.getString("access_token");
			return access_token;
		} catch (Exception e) {
			System.err.printf("获取token失败！");
			e.printStackTrace(System.err);
		}
		return null;
	}
	public String get_text(String content,String url,String accessToken)
	{
		String param;
		String data;
		try {
			//设置请求的编码
			param = "content="+URLEncoder.encode(content,"UTF-8");
			//发送并取得结果
			data = HttpUtil.post(url, accessToken, param);
			System.out.println(data);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return "";
	}