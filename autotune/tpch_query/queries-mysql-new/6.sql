select current_timestamp(6) into @query_start;
set @query_name='6.sql';
SELECT   SUM(L_EXTENDEDPRICE * L_DISCOUNT) AS REVENUE  FROM   LINEITEM  WHERE   L_SHIPDATE >= DATE '1994-01-01'   AND L_SHIPDATE < DATE '1994-01-01' + INTERVAL '1' YEAR   AND L_DISCOUNT BETWEEN 0.06 - 0.01 AND 0.06 + 0.01   AND L_QUANTITY < 24; 
set @query_time_ms= timestampdiff(microsecond, @query_start, current_timestamp(6))/1000;
SELECT @query_name, @query_time_ms;