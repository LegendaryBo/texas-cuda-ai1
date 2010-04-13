package pl.wroc.uni.ii.evolution.distribution2;

import javax.servlet.http.HttpServletResponse;

import org.mortbay.jetty.testing.HttpTester;
import org.mortbay.jetty.testing.ServletTester;

import pl.wroc.uni.ii.evolution.distribution2.server.EvSimpleDistributionServlet;

import junit.framework.TestCase;

/**
 * JUnit TestCase class, testing the EvSimpleDistributionServlet
 * in a dummy manner. Right now, this test is not checking anything.
 * 
 * @author Karol Asgaroth Stosiek (karol.stosiek@gmail.com)
 */
public class EvSimpleDistributionServletTest extends TestCase {
  
  /**
   * this object acts as a container for servlets.
   */
  private ServletTester tester;
  
  /**
   * Sets up the environment for servlet tests.
   * Runs the servlet server.
   */
  @Override
  public void setUp() {  
    this.tester = new ServletTester();
    this.tester.setContextPath("/");
    this.tester.addServlet(
        EvSimpleDistributionServlet.class,
        "/*");
    
    try {
      this.tester.start();
    } catch (Exception e) {
      System.err.println("Failed to start the servlet tester.");
    }
  }
  
  /**
   * Testing, whether POST request are handled.
   */
  public void testDoPostRequest() {
    HttpTester request = new HttpTester();
    HttpTester response = new HttpTester();
    
    request.setMethod("POST");
    request.setHeader("content-type", "application/octet-stream");
    request.setURI("/");
    request.setVersion("HTTP/1.0");

    try {
      response.parse(
          tester.getResponses(
              request.generate()));
    } catch (Exception e) {
      System.err.println("Failed to parse the response.");
    }

    /* we expect servlet to respond with BAD REQUEST 
     * status code, since the content of the request
     * is empty. */
    assertEquals(HttpServletResponse.SC_BAD_REQUEST, 
        response.getStatus());
  }

  /**
   * testing, whether GET requests are handled.
   */
  public void testDoGetRequest() {
    HttpTester request = new HttpTester();
    HttpTester response = new HttpTester();
      
    request.setMethod("GET");
    request.addHeader("amount", "5");
    request.setURI("/");
    request.setContent("");
    request.setVersion("HTTP/1.0");
    
    try {
      response.parse(
          tester.getResponses(
              request.generate()));
    } catch (Exception e) {
      System.err.println("Failed to parse the response.");
    }
    
    assertEquals(HttpServletResponse.SC_OK, 
        response.getStatus());
  }
  
  /**
   * Called, when testing is finished. 
   * Stops the mock-server.
   */
  @Override
  public void tearDown() {
    try {
      this.tester.stop();
    } catch (Exception e) {
      System.err.println("Failed to stop the servlet tester.");
      e.printStackTrace();
    }
  }
}
