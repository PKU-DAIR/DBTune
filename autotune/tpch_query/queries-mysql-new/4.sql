select current_timestamp(6) into @query_start;
set @query_name='4.sql';
SELECT O_ORDERPRIORITY, COUNT(*) AS ORDER_COUNT FROM ORDERS WHERE O_ORDERDATE >= DATE '1995-01-01' AND O_ORDERDATE < DATE '1995-01-01' + INTERVAL '3' MONTH AND EXISTS (SELECT * FROM LINEITEM WHERE L_ORDERKEY = O_ORDERKEY AND L_COMMITDATE < L_RECEIPTDATE) GROUP BY O_ORDERPRIORITY ORDER BY O_ORDERPRIORITY;
set @query_time_ms= timestampdiff(microsecond, @query_start, current_timestamp(6))/1000;
SELECT @query_name, @query_time_ms;