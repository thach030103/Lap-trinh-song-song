<h4>Câu 1: (HW0_P1.cu)</h4>
<h4>Viết hàm và thử nghiệm in ra các thông tin của card màn hình như sau:</h4>
<ul>
  <li>GPU card’s name</li>
  <li>GPU computation capabilities</li>
  <li>Maximum number of block dimensions</li>
  <li>Maximum number of grid dimensions</li>
  <li>Maximum size of GPU memory</li>
  <li>Amount of constant and share memory</li>
  <li>Warp size</li>
</ul>
<h4>Câu 2: (HW0_P2.cu)</h4>
<h4>Viết chương trình cộng hai vector. Tuy nhiên, mỗi thread sẽ thực hiện hai phép tính cộng trên hai phần tử của mảng thay vì một phần tử như trên</h4>
<h4>Version 1:</h4>
<p>Mỗi thread block xử lý 2 * blockDim.x phần tử liên tiếp. Tất cả các thread trong mỗi block sẽ xử lý blockDim.x phần tử đầu mảng, mỗi thread xử lý một phần tử. Sau đó tất cả các thread sẽ chuyển sang blockDim.x phần tử sau của mảng, mỗi thread xử lý một phần tử.</p>
<p>Gọi in1, in2 là hai mảng đầu vào. Out là mảng đầu ra.</p>
<ul>
  <p>Thread 0: sẽ tính out[0] = in1[0] + in2[0] và out[4] = in1[4] + in2[4]</p>
  <p>Thread 1: sẽ tính out[1] = in1[1] + in2[1] và out[5] = in1[5] + in2[5]</p>
  <p>Thread 2: sẽ tính out[2] = in1[2] + in2[2] và out[4] = in1[6] + in2[6]</p>
  <p>Thread 3: sẽ tính out[3] = in1[3] + in2[3] và out[4] = in1[7] + in2[7]</p>
</ul>
<h4>Version 2:</h4>
<p>Mỗi thread block xử lý 2 * blockDim.x phần tử liên tiếp. Mỗi thread sẽ xử lý 2 phần tử liên tiếp nhau trong mảng</p>
<p>Gọi in1, in2 là hai mảng đầu vào. Out là mảng đầu ra.</p>
<ul>
  <p>Thread 0: sẽ tính out[0] = in1[0] + in2[0] và out[1] = in1[1] + in2[1]</p>
  <p>Thread 1: sẽ tính out[2] = in1[2] + in2[2] và out[3] = in1[3] + in2[3]</p>
  <p>Thread 2: sẽ tính out[4] = in1[4] + in2[4] và out[5] = in1[5] + in2[5]</p>
  <p>Thread 3: sẽ tính out[6] = in1[6] + in2[6] và out[7] = in1[7] + in2[7]</p>
</ul>
