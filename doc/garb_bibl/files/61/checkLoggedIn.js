/* Not centrally logged in */
(function(){var t=new Date();t.setTime(t.getTime()+86400000);try{localStorage.setItem('CentralAuthAnon',t.getTime());}catch(e){document.cookie='CentralAuthAnon=1; expires='+t.toGMTString()+'; path=/';}}());