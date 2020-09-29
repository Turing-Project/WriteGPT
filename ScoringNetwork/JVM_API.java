import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;
import java.util.Map;

/**
 * http请求
 * @author 32249
 * @直接使用训练好的DNN评分模型做判断
 */
public class HttpUtil {
	// 发送请求
	/*
	 * @发送消息
	 * @return 结果
	 * 
	 * */
    public static String post(String requestUrl, String accessToken, String params)
            throws Exception 
    {
    	//设置contentType
        String contentType = "application/x-www-form-urlencoded";
        return HttpUtil.post(requestUrl, accessToken, contentType, params);
    }
 
    
    
    public static String post(String requestUrl, String accessToken, String contentType, String params)
            throws Exception
    {
    	//设置编码格式为UTF-8
        String encoding = "UTF-8";
        if (requestUrl.contains("nlp")) 
        {
        	//如果向使用NLP，就要使用GBK编码，这个是百度api是必须要求的
            encoding = "GBK";
        }
        //返回
        return HttpUtil.post(requestUrl, accessToken, contentType, params, encoding);
    }
 
    public static String postGeneralUrl(String generalUrl, String contentType, String params, String encoding)
            throws Exception 
    {
    	//构造URL
        URL url = new URL(generalUrl);
        //构造HttpURLConnection
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("POST");
        //设置Content-Type
        connection.setRequestProperty("Content-Type", contentType);
        //设置连接信息
        connection.setRequestProperty("Connection", "Keep-Alive");
        connection.setUseCaches(false);
 
        connection.setDoOutput(true);
        connection.setDoInput(true);
        //发送请求请求内容
        DataOutputStream out = new DataOutputStream(connection.getOutputStream());
        out.write(params.getBytes(encoding));
        out.flush();
        //关闭
        out.close();
        //连接
        connection.connect();
        //将请求的头打印出来
        Map<String, List<String>> headers = connection.getHeaderFields();
        for (String key : headers.keySet()) {
            System.err.println(key + "--->" + headers.get(key));
        }
        
        BufferedReader in = null;
        //得到结果
        in = new BufferedReader(new InputStreamReader(connection.getInputStream(), encoding));
        String result = "";
        String getLine;
        //将结果取出来
        while ((getLine = in.readLine()) != null) {
            result += getLine;
        }
        in.close();
        //将结果打印出来，返回结果
        System.err.println("result:" + result);
        //返回结果
        return result;
    }
    
    /*
     * @
     * 
     * */
    public static String post(String requestUrl, String accessToken, String contentType, String params, String encoding)
            throws Exception {
    	//构造请求url，将请求信息写入url中
        String url = requestUrl + "?access_token=" + accessToken;
        //返回
        return HttpUtil.postGeneralUrl(url, contentType, params, encoding);
    }